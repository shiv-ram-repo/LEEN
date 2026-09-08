"""
V2 adapter: builds LocalGraphV2 with separate geometry and ESM features.

ESM features are looked up from a precomputed cache keyed by sequence hash,
so it works across Megascale and all benchmarks without name-matching.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from scipy.spatial.distance import cdist

from leen.data.featurizer import (
    AA_1_ORDER,
    physchem_vector,
    compute_backbone_dihedrals,
    encode_dihedrals,
)
from leen.models.leen_v2 import LocalGraphV2

ALPHABET_21 = "ACDEFGHIKLMNPQRSTVWYX"
LEEN_AA_ORDER = AA_1_ORDER

_map = []
for aa in ALPHABET_21:
    _map.append(LEEN_AA_ORDER.index(aa) if aa in LEEN_AA_ORDER else 20)
ALPHA21_TO_LEEN = np.array(_map, dtype=np.int64)


def masked_seq_key(S_np, pos: int) -> str:
    """Key by the sequence with the mutation site replaced by a sentinel 'Z'.
    WT and mutant sequences differ ONLY at the site, so masking it collapses them to ONE key
    -> the masked ESM embedding is shared by both directions -> exact independent anti-symmetry."""
    seq = "".join(ALPHABET_21[i] if i < 21 else "X" for i in S_np)
    if 0 <= pos < len(seq):
        seq = seq[:pos] + "Z" + seq[pos + 1:]
    return hashlib.md5(seq.encode()).hexdigest()


def _seq_key(S_np: np.ndarray) -> str:
    """Reconstruct sequence from S tensor and hash it."""
    seq = "".join(ALPHABET_21[i] if i < 21 else "X" for i in S_np)
    clean = seq.replace("-", "").replace("X", "")
    return hashlib.md5(clean.encode()).hexdigest()


def _rbf(d, D_max=10.0, n=13):
    c = np.linspace(0, D_max, n)
    return np.exp(-(1.0 / (D_max / n) ** 2) * (d - c) ** 2).astype(np.float32)


def _node_feats(ca, n_c, c_c, S_np, L):
    phi, psi = compute_backbone_dihedrals(
        [n_c[i] for i in range(L)],
        [ca[i] for i in range(L)],
        [c_c[i] for i in range(L)],
    )
    dihed = encode_dihedrals(phi, psi)
    feats = []
    for i in range(L):
        aa_1 = ALPHABET_21[int(S_np[i])] if int(S_np[i]) < 21 else "X"
        ph = physchem_vector(aa_1)
        rp = np.array([i / max(L - 1, 1)], dtype=np.float32)
        pad = np.array([0.0], dtype=np.float32)
        feats.append(np.concatenate([ph, dihed[i], rp, pad]))
    return np.stack(feats)


def protein_to_graphs_v2_masked(
    protein: dict,
    esm_cache: dict | None = None,
    masked_esm_cache: dict | None = None,
    env_radius: float = 10.0,
    edge_cutoff: float = 8.0,
    max_neighbors: int = 64,
) -> list[LocalGraphV2]:
    """Convert protein dict → list of LocalGraphV2 with separate ESM features."""

    X = protein["X"]
    S = protein["S"]
    # tolerate both torch tensors (eval path) and numpy arrays (training path)
    X_np = X.cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    S_np = S.cpu().numpy() if torch.is_tensor(S) else np.asarray(S)
    if X_np.ndim == 4:
        X_np = X_np[0]
    if S_np.ndim == 2:
        S_np = S_np[0]
    L = X_np.shape[0]

    n_coords = X_np[:, 0, :]
    ca_coords = X_np[:, 1, :]
    c_coords = X_np[:, 2, :]

    geo_feats = _node_feats(ca_coords, n_coords, c_coords, S_np, L)
    aa_full = ALPHA21_TO_LEEN[np.clip(S_np, 0, 20)]

    # ESM lookup by sequence hash
    esm_emb = None
    if esm_cache is not None:
        key = _seq_key(S_np)
        if key in esm_cache:
            esm_emb = esm_cache[key]
            if esm_emb.shape[0] < L:
                esm_emb = torch.cat([esm_emb, torch.zeros(L - esm_emb.shape[0], esm_emb.shape[1])])
            elif esm_emb.shape[0] > L:
                esm_emb = esm_emb[:L]

    mut_ids = protein["mut_ids"]
    append_tensors = protein["append_tensors"]
    ddG = protein["ddG"]

    graphs = []

    for mi in range(len(mut_ids)):
        mut_pos = mut_ids[mi]
        if mut_pos < 0 or mut_pos >= L:
            continue

        wt_aa = ALPHA21_TO_LEEN[int(append_tensors[mi, :21].argmax().item())]
        mt_aa = ALPHA21_TO_LEEN[int(append_tensors[mi, 21:].argmax().item())]
        if wt_aa >= 20 or mt_aa >= 20:
            continue

        _ddg_ndim = ddG.dim() if torch.is_tensor(ddG) else np.asarray(ddG).ndim
        ddg_val = float(ddG[mi]) if _ddg_ndim == 1 else float(ddG[mi, 0])

        # Local neighborhood
        mut_ca = ca_coords[mut_pos]
        dists = np.linalg.norm(ca_coords - mut_ca, axis=-1)
        local_idx = np.where(dists <= env_radius)[0]

        if len(local_idx) > max_neighbors:
            local_idx = np.sort(local_idx[np.argsort(dists[local_idx])][:max_neighbors])
        if mut_pos not in local_idx:
            local_idx = np.sort(np.append(local_idx, mut_pos))

        N = len(local_idx)
        g2l = {int(g): l for l, g in enumerate(local_idx)}
        mut_local = g2l[mut_pos]

        nf = geo_feats[local_idx].copy()
        coords = ca_coords[local_idx] - mut_ca
        aa_loc = aa_full[local_idx].copy()

        # ---- TRUE ANTI-SYMMETRY: blind the mutation-site identity across ALL channels ----
        # Only the single mutated residue's identity is hidden; everything else keeps full info.
        #   channel 1: node AA embedding index at the site -> unknown (20)
        aa_loc[mut_local] = 20
        #   channel 2: physicochemical vector (leading 7 dims of the site node feature) -> 0
        nf[mut_local, :7] = 0.0
        #   channel 3: ESM. Use the FULL site-masked-sequence embedding for the WHOLE neighborhood,
        #   so neighbor rows are identity-independent too (ESM is contextual). WT and mutant inputs
        #   -> identical site-masked sequence -> identical esm_loc -> exact anti-symmetry.
        esm_loc = None
        masked_full = None
        if masked_esm_cache is not None:
            masked_full = masked_esm_cache.get(masked_seq_key(S_np, int(mut_pos)))
        if masked_full is not None:
            mf = masked_full if torch.is_tensor(masked_full) else torch.as_tensor(masked_full)
            esm_loc = mf[local_idx].float().clone()    # [N, 1280] from masked seq (cast fp16->fp32)
        elif esm_emb is not None:
            # fallback (masked emb missing): at least blank the site row (weaker, not exact)
            esm_loc = esm_emb[local_idx].clone()
            esm_loc[mut_local] = 0.0


        # Edges
        dm = cdist(ca_coords[local_idx], ca_coords[local_idx])
        src, dst, ef = [], [], []
        for i in range(N):
            for j in range(N):
                if i != j and dm[i, j] <= edge_cutoff:
                    src.append(i)
                    dst.append(j)
                    d = dm[i, j]
                    ss = abs(int(local_idx[i]) - int(local_idx[j]))
                    ef.append(np.concatenate([_rbf(d, edge_cutoff), [ss / 20, float(ss == 1), d / edge_cutoff]]))

        if not src:
            for j in range(N):
                if j != mut_local:
                    for a, b in [(mut_local, j), (j, mut_local)]:
                        src.append(a); dst.append(b)
                        d = dm[a, b]
                        ef.append(np.concatenate([_rbf(d, 20.0), [0, 0, d / 20]]))

        graphs.append(LocalGraphV2(
            node_features=torch.tensor(nf, dtype=torch.float32),
            coords=torch.tensor(coords, dtype=torch.float32),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_attr=torch.tensor(np.array(ef), dtype=torch.float32),
            aa_indices=torch.tensor(aa_loc, dtype=torch.long),
            esm_features=esm_loc,
            mut_pos=torch.tensor([mut_local], dtype=torch.long),
            wt_aa=torch.tensor([wt_aa], dtype=torch.long),
            mut_aa=torch.tensor([mt_aa], dtype=torch.long),
            ddg=torch.tensor([ddg_val], dtype=torch.float32),
            batch=torch.zeros(N, dtype=torch.long),
        ))

    return graphs


def collate_v2(graphs: list[LocalGraphV2]) -> LocalGraphV2:
    """Batch LocalGraphV2 instances."""
    nf, co, ei, ea = [], [], [], []
    aa, esm, mp, wt, mt, dd, bt = [], [], [], [], [], [], []
    off = 0
    has_esm = graphs[0].esm_features is not None

    for i, g in enumerate(graphs):
        N = g.node_features.size(0)
        nf.append(g.node_features)
        co.append(g.coords)
        aa.append(g.aa_indices)
        ei.append(g.edge_index + off)
        if g.edge_attr is not None:
            ea.append(g.edge_attr)
        if has_esm and g.esm_features is not None:
            esm.append(g.esm_features)
        mp.append(g.mut_pos + off)
        wt.append(g.wt_aa)
        mt.append(g.mut_aa)
        if g.ddg is not None:
            dd.append(g.ddg)
        bt.append(torch.full((N,), i, dtype=torch.long))
        off += N

    return LocalGraphV2(
        node_features=torch.cat(nf),
        coords=torch.cat(co),
        edge_index=torch.cat(ei, dim=1),
        edge_attr=torch.cat(ea) if ea else None,
        aa_indices=torch.cat(aa),
        esm_features=torch.cat(esm) if esm else None,
        mut_pos=torch.cat(mp),
        wt_aa=torch.cat(wt),
        mut_aa=torch.cat(mt),
        ddg=torch.cat(dd) if dd else None,
        batch=torch.cat(bt),
    )
