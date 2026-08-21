"""
LEEN: Local Equivariant Energy Network.

Core idea: instead of regressing ΔΔG directly, learn a per-residue energy
function E(amino_acid, structural_context) and compute

    ΔΔG = E(mutant_aa, env) − E(wildtype_aa, env)

This gives exact antisymmetry by construction:
    ΔΔG(wt→mut) = E(mut) − E(wt) = −(E(wt) − E(mut)) = −ΔΔG(mut→wt)

No Siamese loss, no data augmentation, no architectural tricks needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from leen.models.egnn import EGNN


# ──────────────────────────────────────────────────────────────
# Data container for a single local graph (or batched)
# ──────────────────────────────────────────────────────────────

@dataclass
class LocalGraph:
    """Container for a local protein microenvironment graph.

    All tensors are on the same device. For batched graphs,
    node-level tensors are concatenated along dim 0 and
    `batch` maps each node to its sample index.
    """

    node_features: torch.Tensor   # [N_total, feat_dim]
    coords: torch.Tensor          # [N_total, 3]
    edge_index: torch.Tensor      # [2, E_total]
    edge_attr: torch.Tensor | None  # [E_total, edge_dim] or None

    aa_indices: torch.Tensor      # [N_total] amino acid index per node (0–19)
    mut_pos: torch.Tensor         # [B] index into node array for mutation site
    wt_aa: torch.Tensor           # [B] wildtype amino acid index (0–19)
    mut_aa: torch.Tensor          # [B] mutant amino acid index (0–19)
    ddg: torch.Tensor | None      # [B] experimental ΔΔG (None at inference)
    batch: torch.Tensor           # [N_total] sample index per node

    def to(self, device: torch.device) -> "LocalGraph":
        fields = {}
        for k, v in self.__dict__.items():
            fields[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        return LocalGraph(**fields)


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

class LEEN(nn.Module):
    """Local Equivariant Energy Network.

    Architecture:
        1. Amino acid embedding + structural features → initial node repr
        2. EGNN encoder on local graph → context-aware node features
        3. Energy head at mutation site → 20-dim energy vector
        4. ΔΔG = E[mut_aa] − E[wt_aa]
    """

    NUM_AA = 20  # standard amino acids

    def __init__(
        self,
        input_node_dim: int,
        aa_embed_dim: int = 64,
        node_dim: int = 128,
        edge_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 6,
        update_coords: bool = True,
        dropout: float = 0.1,
        energy_hidden_dim: int = 256,
    ):
        super().__init__()

        # Learnable amino acid embedding (added to node features)
        self.aa_embedding = nn.Embedding(self.NUM_AA + 1, aa_embed_dim)  # +1 for mask/pad

        # EGNN backbone encoder
        self.encoder = EGNN(
            input_dim=input_node_dim + aa_embed_dim,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            update_coords=update_coords,
            dropout=dropout,
        )

        # Energy prediction head
        # Maps the mutation-site representation to 20 energy scalars,
        # one per amino acid type.
        self.energy_head = nn.Sequential(
            nn.Linear(node_dim, energy_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(energy_hidden_dim, energy_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(energy_hidden_dim, self.NUM_AA),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize energy head with small weights for stable early training."""
        for m in self.energy_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, graph: LocalGraph) -> torch.Tensor:
        """Run encoder and return node features at the mutation site.

        Returns:
            h_mut: [B, node_dim] — encoded representation at mutation positions.
        """
        # Combine structural features with amino acid identity
        aa_emb = self.aa_embedding(graph.aa_indices)  # [N, aa_embed_dim]
        x_in = torch.cat([graph.node_features, aa_emb], dim=-1)

        # EGNN forward
        h, _ = self.encoder(x_in, graph.coords, graph.edge_index, graph.edge_attr)

        # Gather mutation-site representations
        h_mut = h[graph.mut_pos]  # [B, node_dim]
        return h_mut

    def predict_energy(self, graph: LocalGraph) -> torch.Tensor:
        """Predict the full 20-dimensional energy vector at the mutation site.

        Returns:
            energy: [B, 20] — one energy scalar per amino acid type.
        """
        h_mut = self.encode(graph)
        energy = self.energy_head(h_mut)  # [B, 20]
        return energy

    def forward(self, graph: LocalGraph) -> dict[str, torch.Tensor]:
        """Predict ΔΔG via energy difference.

        Returns dict with:
            ddg_pred: [B] — predicted ΔΔG = E(mut) − E(wt)
            energy:   [B, 20] — full energy vector (for analysis)
            h_mut:    [B, node_dim] — mutation-site representation (for OOD loss)
        """
        h_mut = self.encode(graph)
        energy = self.energy_head(h_mut)  # [B, 20]

        # Gather wt and mut energies
        e_wt = energy.gather(1, graph.wt_aa.unsqueeze(1)).squeeze(1)   # [B]
        e_mut = energy.gather(1, graph.mut_aa.unsqueeze(1)).squeeze(1)  # [B]

        ddg_pred = e_mut - e_wt  # exact antisymmetry

        return {
            "ddg_pred": ddg_pred,
            "energy": energy,
            "h_mut": h_mut,
        }

    def predict_with_perturbation(
        self, graph: LocalGraph, sigma: float = 0.2
    ) -> dict[str, torch.Tensor]:
        """Forward pass with OOD-margin noise on the encoded representation.

        Adds Gaussian noise to h_mut, re-runs the energy head, and returns
        both clean and perturbed predictions for the OOD consistency loss.
        """
        h_mut = self.encode(graph)
        energy_clean = self.energy_head(h_mut)

        # Perturb
        noise = torch.randn_like(h_mut) * sigma
        h_perturbed = h_mut + noise
        energy_perturbed = self.energy_head(h_perturbed)

        # ΔΔG from clean
        e_wt = energy_clean.gather(1, graph.wt_aa.unsqueeze(1)).squeeze(1)
        e_mut = energy_clean.gather(1, graph.mut_aa.unsqueeze(1)).squeeze(1)
        ddg_clean = e_mut - e_wt

        # ΔΔG from perturbed
        e_wt_p = energy_perturbed.gather(1, graph.wt_aa.unsqueeze(1)).squeeze(1)
        e_mut_p = energy_perturbed.gather(1, graph.mut_aa.unsqueeze(1)).squeeze(1)
        ddg_perturbed = e_mut_p - e_wt_p

        return {
            "ddg_pred": ddg_clean,
            "ddg_perturbed": ddg_perturbed,
            "energy": energy_clean,
            "h_mut": h_mut,
        }
