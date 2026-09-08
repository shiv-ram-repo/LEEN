"""
Adapter between the existing protstab_data pipeline and LEEN.

Converts the protein dict returned by MegaScaleDataset / ddgBenchDataset
into a list of LocalGraph objects for the LEEN model.

The protein dict contains:
    X               : [1, L, 4, 3]  backbone atoms (N, CA, C, O)
    S               : [1, L]         sequence indices (ALPHABET_21)
    mut_ids         : list[int]      mutation positions (0-indexed)
    ddG             : [N_muts, 1]    experimental ΔΔG
    append_tensors  : [N_muts, 42]   [wt_onehot_21 | mut_onehot_21]
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.distance import cdist

from leen.models.energy_model import LocalGraph
from leen.data.featurizer import (
    AA_1_ORDER,
    physchem_vector,
    compute_backbone_dihedrals,
    encode_dihedrals,
)

# protstab_data uses this 21-char alphabet (X = unknown)
ALPHABET_21 = "ACDEFGHIKLMNPQRSTVWYX"
# LEEN uses the 20 standard amino acids (same order, minus X)
LEEN_AA_ORDER = AA_1_ORDER  # "ACDEFGHIKLMNPQRSTVWY"

# Mapping from ALPHABET_21 index → LEEN 0–19 index
# X (index 20 in ALPHABET_21) maps to 20 (the padding slot in nn.Embedding(21, ...))
_ALPHA21_TO_LEEN = []
for aa in ALPHABET_21:
    if aa in LEEN_AA_ORDER:
        _ALPHA21_TO_LEEN.append(LEEN_AA_ORDER.index(aa))
    else:
        _ALPHA21_TO_LEEN.append(20)  # padding index, NOT -1
ALPHA21_TO_LEEN = np.array(_ALPHA21_TO_LEEN, dtype=np.int64)


def _rbf_encode(distance: float, D_max: float = 10.0, num_rbf: int = 13) -> np.ndarray:
    """Radial basis function distance encoding."""
    centers = np.linspace(0.0, D_max, num_rbf)
    gamma = 1.0 / (D_max / num_rbf) ** 2
    return np.exp(-gamma * (distance - centers) ** 2).astype(np.float32)


def _build_node_features(
    ca_coords: np.ndarray,
    n_coords: np.ndarray,
    c_coords: np.ndarray,
    seq_indices: np.ndarray,
    L: int,
) -> np.ndarray:
    """Build per-residue feature vectors [L, 13].

    Features: physicochemical [7] + dihedral sin/cos [4] + relative_pos [1] + padding [1].
    """
    # Backbone dihedrals
    phi, psi = compute_backbone_dihedrals(
        [n_coords[i] for i in range(L)],
        [ca_coords[i] for i in range(L)],
        [c_coords[i] for i in range(L)],
    )
    dihed = encode_dihedrals(phi, psi)  # [L, 4]

    features = []
    for i in range(L):
        aa_idx = int(seq_indices[i])
        aa_1 = ALPHABET_21[aa_idx] if aa_idx < len(ALPHABET_21) else "X"
        phys = physchem_vector(aa_1)      # [7]
        rel_pos = np.array([i / max(L - 1, 1)], dtype=np.float32)
        pad = np.array([0.0], dtype=np.float32)
        features.append(np.concatenate([phys, dihed[i], rel_pos, pad]))

    return np.stack(features)  # [L, 13]


def protein_dict_to_local_graphs(
    protein: dict,
    env_radius: float = 10.0,
    edge_cutoff: float = 8.0,
    max_neighbors: int = 64,
) -> list[LocalGraph]:
    """Convert a protstab_data protein dict into a list of LEEN LocalGraphs.

    One LocalGraph per mutation in the protein's mut_ids list.

    Args:
        protein: dict from MegaScaleDataset/ddgBenchDataset __getitem__.
        env_radius: Å radius for local environment.
        edge_cutoff: Å cutoff for graph edges.
        max_neighbors: max residues in local graph.

    Returns:
        List of LocalGraph objects, one per mutation. Empty list if no valid
        mutations could be built.
    """
    X = protein["X"]                # [1, L, 4, 3] or [L, 4, 3]
    S = protein["S"]                # [1, L] or [L]
    mut_ids = protein["mut_ids"]
    ddG = protein["ddG"]            # [N, 1]
    append_tensors = protein["append_tensors"]  # [N, 42]

    # Squeeze batch dim if present
    if X.dim() == 4:
        X = X.squeeze(0)  # [L, 4, 3]
    if S.dim() == 2:
        S = S.squeeze(0)  # [L]

    X_np = X.cpu().numpy()
    S_np = S.cpu().numpy()
    L = X_np.shape[0]

    # Backbone atom coordinates
    n_coords = X_np[:, 0, :]   # N atoms
    ca_coords = X_np[:, 1, :]  # CA atoms
    c_coords = X_np[:, 2, :]   # C atoms

    # Build full-protein node features once
    node_features_full = _build_node_features(ca_coords, n_coords, c_coords, S_np, L)

    # Map S indices from ALPHABET_21 to LEEN's 0–19
    aa_full = ALPHA21_TO_LEEN[np.clip(S_np, 0, 20)]  # [L]

    graphs = []
    n_muts = len(mut_ids)

    for mi in range(n_muts):
        mut_pos = mut_ids[mi]
        if mut_pos < 0 or mut_pos >= L:
            continue

        # Extract wt/mut amino acid from append_tensors
        wt_onehot = append_tensors[mi, :21]
        mt_onehot = append_tensors[mi, 21:]
        wt_aa_21 = int(wt_onehot.argmax().item())
        mt_aa_21 = int(mt_onehot.argmax().item())

        # Map to LEEN indices (skip X/unknown = 20)
        wt_aa = ALPHA21_TO_LEEN[wt_aa_21]
        mt_aa = ALPHA21_TO_LEEN[mt_aa_21]
        if wt_aa >= 20 or mt_aa >= 20:
            continue  # skip unknown amino acids

        # ΔΔG target
        ddg_val = ddG[mi].item() if ddG.dim() == 1 else ddG[mi, 0].item()

        # ── Local neighborhood ──
        mut_ca = ca_coords[mut_pos]
        distances = np.linalg.norm(ca_coords - mut_ca, axis=-1)
        local_mask = distances <= env_radius
        local_indices = np.where(local_mask)[0]

        if len(local_indices) > max_neighbors:
            sorted_idx = local_indices[np.argsort(distances[local_indices])]
            local_indices = np.sort(sorted_idx[:max_neighbors])

        if mut_pos not in local_indices:
            local_indices = np.sort(np.append(local_indices, mut_pos))

        num_local = len(local_indices)
        global_to_local = {int(g): l for l, g in enumerate(local_indices)}
        mut_local = global_to_local[mut_pos]

        # Node features and coordinates (centered at mutation site)
        nf = node_features_full[local_indices]
        coords = ca_coords[local_indices] - mut_ca
        aa_local = aa_full[local_indices]

        # ── Edges ──
        local_ca = ca_coords[local_indices]
        dm = cdist(local_ca, local_ca)
        src, dst, edge_feats = [], [], []

        for i in range(num_local):
            for j in range(num_local):
                if i == j:
                    continue
                if dm[i, j] <= edge_cutoff:
                    src.append(i)
                    dst.append(j)
                    d = dm[i, j]
                    seq_sep = abs(int(local_indices[i]) - int(local_indices[j]))
                    is_bonded = 1.0 if seq_sep == 1 else 0.0
                    rbf = _rbf_encode(d, D_max=edge_cutoff)
                    edge_feats.append(np.concatenate([
                        rbf, [seq_sep / 20.0, is_bonded, d / edge_cutoff]
                    ]))

        if not src:
            # Fallback: connect mutation site to everything
            for j in range(num_local):
                if j != mut_local:
                    for a, b in [(mut_local, j), (j, mut_local)]:
                        src.append(a)
                        dst.append(b)
                        d = dm[a, b]
                        rbf = _rbf_encode(d, D_max=20.0)
                        edge_feats.append(np.concatenate([rbf, [0.0, 0.0, d / 20.0]]))

        graph = LocalGraph(
            node_features=torch.tensor(nf, dtype=torch.float32),
            coords=torch.tensor(coords, dtype=torch.float32),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_attr=torch.tensor(np.array(edge_feats), dtype=torch.float32),
            aa_indices=torch.tensor(aa_local, dtype=torch.long),
            mut_pos=torch.tensor([mut_local], dtype=torch.long),
            wt_aa=torch.tensor([wt_aa], dtype=torch.long),
            mut_aa=torch.tensor([mt_aa], dtype=torch.long),
            ddg=torch.tensor([ddg_val], dtype=torch.float32),
            batch=torch.zeros(num_local, dtype=torch.long),
        )
        graphs.append(graph)

    return graphs


def collate_local_graphs(graphs: list[LocalGraph]) -> LocalGraph:
    """Batch multiple LocalGraphs into one, offsetting indices."""
    node_feats, coords, edge_idxs, edge_attrs = [], [], [], []
    aa_idxs, mut_pos, wt_aas, mut_aas, ddgs, batches = [], [], [], [], [], []
    offset = 0

    for i, g in enumerate(graphs):
        N = g.node_features.size(0)
        node_feats.append(g.node_features)
        coords.append(g.coords)
        aa_idxs.append(g.aa_indices)
        edge_idxs.append(g.edge_index + offset)
        if g.edge_attr is not None:
            edge_attrs.append(g.edge_attr)
        mut_pos.append(g.mut_pos + offset)
        wt_aas.append(g.wt_aa)
        mut_aas.append(g.mut_aa)
        if g.ddg is not None:
            ddgs.append(g.ddg)
        batches.append(torch.full((N,), i, dtype=torch.long))
        offset += N

    return LocalGraph(
        node_features=torch.cat(node_feats),
        coords=torch.cat(coords),
        edge_index=torch.cat(edge_idxs, dim=1),
        edge_attr=torch.cat(edge_attrs) if edge_attrs else None,
        aa_indices=torch.cat(aa_idxs),
        mut_pos=torch.cat(mut_pos),
        wt_aa=torch.cat(wt_aas),
        mut_aa=torch.cat(mut_aas),
        ddg=torch.cat(ddgs) if ddgs else None,
        batch=torch.cat(batches),
    )
