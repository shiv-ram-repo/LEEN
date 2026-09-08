"""
Amino acid encoding and residue-level feature computation for LEEN.
Constants + backbone dihedral computation used by the adapter.
"""

from __future__ import annotations
import numpy as np

AA_1_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_1_ORDER)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}

PHYSCHEM = {
    "A": [ 0.62, -0.89,  0.0,  0.0, 0.0, 0.0, 0.0],
    "R": [-2.53,  0.70,  1.0,  1.0, 0.0, 1.0, 0.0],
    "N": [-0.78, -0.35,  0.0,  1.0, 0.0, 1.0, 1.0],
    "D": [-0.90, -0.48, -1.0,  1.0, 0.0, 0.0, 1.0],
    "C": [ 0.29, -0.41,  0.0,  0.0, 0.0, 0.0, 0.0],
    "Q": [-0.85,  0.09,  0.0,  1.0, 0.0, 1.0, 1.0],
    "E": [-0.74,  0.11, -1.0,  1.0, 0.0, 0.0, 1.0],
    "G": [ 0.48, -1.72,  0.0,  0.0, 0.0, 0.0, 0.0],
    "H": [-0.40,  0.32,  0.5,  1.0, 1.0, 1.0, 1.0],
    "I": [ 1.38,  0.45,  0.0,  0.0, 0.0, 0.0, 0.0],
    "L": [ 1.06,  0.45,  0.0,  0.0, 0.0, 0.0, 0.0],
    "K": [-1.50,  0.56,  1.0,  1.0, 0.0, 1.0, 0.0],
    "M": [ 0.64,  0.38,  0.0,  0.0, 0.0, 0.0, 0.0],
    "F": [ 1.19,  0.89,  0.0,  0.0, 1.0, 0.0, 0.0],
    "P": [ 0.12, -0.27,  0.0,  0.0, 0.0, 0.0, 0.0],
    "S": [-0.18, -0.67,  0.0,  1.0, 0.0, 1.0, 1.0],
    "T": [-0.05, -0.12,  0.0,  1.0, 0.0, 1.0, 1.0],
    "W": [-0.46,  1.40,  0.0,  0.0, 1.0, 1.0, 0.0],
    "Y": [ 0.26,  1.01,  0.0,  1.0, 1.0, 1.0, 1.0],
    "V": [ 1.08,  0.07,  0.0,  0.0, 0.0, 0.0, 0.0],
}

def physchem_vector(aa_1letter: str) -> np.ndarray:
    return np.array(PHYSCHEM.get(aa_1letter, [0.0] * 7), dtype=np.float32)

def _dihedral(p0, p1, p2, p3):
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    n1n, n2n = np.linalg.norm(n1), np.linalg.norm(n2)
    if n1n < 1e-8 or n2n < 1e-8:
        return 0.0
    n1, n2 = n1 / n1n, n2 / n2n
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-8))
    return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))

def compute_backbone_dihedrals(coords_N, coords_CA, coords_C):
    L = len(coords_CA)
    phi, psi = np.zeros(L, np.float32), np.zeros(L, np.float32)
    for i in range(L):
        if i > 0:
            phi[i] = _dihedral(coords_C[i-1], coords_N[i], coords_CA[i], coords_C[i])
        if i < L - 1:
            psi[i] = _dihedral(coords_N[i], coords_CA[i], coords_C[i], coords_N[i+1])
    return phi, psi

def encode_dihedrals(phi, psi):
    return np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=-1).astype(np.float32)
