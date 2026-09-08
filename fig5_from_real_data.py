#!/usr/bin/env python3
"""
Fig 5 (anti-symmetry) from REAL predictions — no synthetic scatter.
Needs: E2 forward+reverse Ssym predictions, and symmetrized-variant predictions.
Ssym-direct and Ssym-inverse are the SAME mutations in opposite directions, so
for each mutation: forward = ssym_direct pred, reverse = ssym_inverse pred.

Run from LEEN_clean:
  python fig5_from_real_data.py
Falls back with a clear message if the prediction files aren't where expected.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def load_dir_rev(cfg_dir):
    """Return (fwd, rev) arrays aligned by mutation for a config's Ssym preds."""
    d=pd.read_csv(f"{cfg_dir}/ssym_direct.csv")
    i=pd.read_csv(f"{cfg_dir}/ssym_inverse.csv")
    # align by row order (both are the same 342 mutations in the pipeline)
    n=min(len(d),len(i))
    return d['pred'].to_numpy()[:n], i['pred'].to_numpy()[:n]

# --- primary E2 (fixed-context): expect small nonzero fwd+rev ---
E2_DIR="runs/leen_preds/E2_seed42"
# --- symmetrized variant: exact. Adjust dir name to your symmetrized preds. ---
SYM_DIR="runs/leen_preds/E2sym_seed42"   # <-- CHANGE if your symmetrized preds live elsewhere

have_e2 = os.path.exists(f"{E2_DIR}/ssym_direct.csv")
have_sym = os.path.exists(f"{SYM_DIR}/ssym_direct.csv")
print("E2 preds:", "found" if have_e2 else f"MISSING at {E2_DIR}")
print("SYM preds:", "found" if have_sym else f"MISSING at {SYM_DIR} (edit SYM_DIR)")

fig, axes = plt.subplots(1,3, figsize=(12,3.6))

# (a) symmetrized: exact
ax=axes[0]
ax.plot([-4,4],[4,-4],'--',color='gray',lw=1)
if have_sym:
    f,r = load_dir_rev(SYM_DIR)
    ax.scatter(f, r, s=6, color="#1f77b4", alpha=0.6)
    viol = np.abs(f + r)
    ax.text(0.05,0.05,f"max viol $\\approx${viol.max():.1e}", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round",fc="#dbeafe",ec="none"))
else:
    ax.text(0.5,0.5,"symmetrized preds\nnot found",ha="center",transform=ax.transAxes)
ax.set_title("(a) Symmetrized LEEN: exact anti-symmetry")
ax.set_xlabel(r"$\Delta\Delta G_{\mathrm{wt\to mut}}$"); ax.set_ylabel(r"$\Delta\Delta G_{\mathrm{mut\to wt}}$")
ax.set_xlim(-4,4); ax.set_ylim(-4,4)

# (b) primary E2: real residual
ax=axes[1]
ax.plot([-4,4],[4,-4],'--',color='gray',lw=1)
if have_e2:
    f,r = load_dir_rev(E2_DIR)
    ax.scatter(f, r, s=6, color="#d62728", alpha=0.55)
    viol=np.abs(f+r)
    ax.text(0.05,0.05,f"mean viol $\\approx${viol.mean():.2f}", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round",fc="#fde2e2",ec="none"))
else:
    ax.text(0.5,0.5,"E2 preds not found",ha="center",transform=ax.transAxes)
ax.set_title("(b) Primary E2: small residual")
ax.set_xlabel(r"$\Delta\Delta G_{\mathrm{wt\to mut}}$")
ax.set_xlim(-4,4); ax.set_ylim(-4,4)

# (c) violation distributions
ax=axes[2]
if have_e2:
    f,r=load_dir_rev(E2_DIR); ax.hist(np.abs(f+r), bins=30, color="#d62728", alpha=0.6, density=True, label="Primary E2")
if have_sym:
    f,r=load_dir_rev(SYM_DIR); ax.hist(np.abs(f+r), bins=30, color="#1f77b4", alpha=0.6, density=True, label="Symmetrized")
ax.set_title("(c) Forward+reverse violation")
ax.set_xlabel(r"$|\Delta\Delta G_{\mathrm{fwd}}+\Delta\Delta G_{\mathrm{rev}}|$"); ax.set_ylabel("Density")
ax.legend(fontsize=8)

plt.tight_layout(); os.makedirs("figs_new",exist_ok=True)
plt.savefig("figs_new/fig5_antisymmetry_REAL.pdf"); plt.close()
print("wrote figs_new/fig5_antisymmetry_REAL.pdf")
print("If SYM_DIR was wrong, find your symmetrized Ssym preds and edit the path, rerun.")
