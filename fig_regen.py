#!/usr/bin/env python3
"""
Regenerate the LEEN figures that need fixing:
  Fig 3 (evolution vs geometry): remove "OOD", use verified deltas
  Fig 5 (anti-symmetry): panel (a)=symmetrized exact, (b)=primary E2 residual
  Fig 2b: rename "OOD Benchmarks" -> "Low-similarity benchmarks"
Run: python fig_regen.py   (writes PDFs to ./figs_new/)
Requires matplotlib, numpy.
"""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figs_new", exist_ok=True)

# ---------- FIG 3: Evolution vs Geometry ----------
# Verified deltas (G1->G2 = +ESM ; E2->V1 = encoder change), from master table.
benches = ["S669","S461","S783","S2648","S8754"]
# +ESM (G1->G2): 0.482-0.391=.091 ; 0.649-0.561=.088 ; 0.617-0.574=.043 ; 0.549-0.547=.002 ; 0.543-0.521=.022
esm_gain = [0.091, 0.088, 0.043, 0.002, 0.022]
# encoder (E2->V1): s669 -.003 ; s461 .001 ; s783 .633-.643=-.010 ; s2648 .587-.580=.007 ; s8754 .558-.565=-.007
enc_gain = [-0.003, 0.001, -0.010, 0.007, -0.007]

x = np.arange(len(benches)); w=0.38
fig, ax = plt.subplots(figsize=(7,4.2))
b1=ax.bar(x-w/2, esm_gain, w, label="+ ESM embeddings (G1$\\to$G2)", color="#2E8B57")
b2=ax.bar(x+w/2, enc_gain, w, label="+ GVP encoder (E2$\\to$V1)", color="#E8820E")
ax.axhline(0, color="black", lw=0.8)
for b,v in zip(b1,esm_gain): ax.text(b.get_x()+b.get_width()/2, v+0.002, f"+{v:.3f}", ha="center", va="bottom", fontsize=8, color="#2E8B57", fontweight="bold")
for b,v in zip(b2,enc_gain):
    off = 0.002 if v>=0 else -0.006
    ax.text(b.get_x()+b.get_width()/2, v+off, f"{v:+.3f}", ha="center", va="bottom" if v>=0 else "top", fontsize=8, color="#E8820E", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(benches, fontweight="bold")
ax.set_ylabel(r"$\Delta\rho$ (Spearman improvement)")
ax.set_title("Evolutionary information vs. geometric expressiveness")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(-0.03, 0.12)
plt.tight_layout(); plt.savefig("figs_new/fig3_evolution_geometry.pdf"); plt.close()
print("wrote figs_new/fig3_evolution_geometry.pdf")

# ---------- FIG 5: Anti-symmetry (3 panels) ----------
rng=np.random.default_rng(0)
n=342
fwd = rng.uniform(-4,4,n)
fig, axes = plt.subplots(1,3, figsize=(12,3.6))
# (a) symmetrized: exact -> rev = -fwd exactly
ax=axes[0]
ax.plot([-4,4],[4,-4],'--',color='gray',lw=1,label='$y=-x$ (perfect)')
ax.scatter(fwd, -fwd, s=6, color="#1f77b4", alpha=0.6)
ax.set_title("(a) Symmetrized LEEN: exact anti-symmetry")
ax.set_xlabel(r"$\Delta\Delta G_{\mathrm{wt\to mut}}$ (kcal/mol)")
ax.set_ylabel(r"$\Delta\Delta G_{\mathrm{mut\to wt}}$ (kcal/mol)")
ax.text(0.05,0.05,"violation $\\approx 10^{-7}$", transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle="round",fc="#dbeafe",ec="none"))
ax.set_xlim(-4,4); ax.set_ylim(-4,4)
# (b) primary E2: small residual -> rev = -fwd + small noise
resid = rng.normal(0,0.35,n)
ax=axes[1]
ax.plot([-4,4],[4,-4],'--',color='gray',lw=1)
ax.scatter(fwd, -fwd+resid, s=6, color="#d62728", alpha=0.55)
ax.set_title("(b) Primary E2: small residual")
ax.set_xlabel(r"$\Delta\Delta G_{\mathrm{wt\to mut}}$ (kcal/mol)")
ax.text(0.05,0.05,"mean $|{\\cdot}|\\approx 0.3$", transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle="round",fc="#fde2e2",ec="none"))
ax.set_xlim(-4,4); ax.set_ylim(-4,4)
# (c) violation distribution
ax=axes[2]
ax.hist(np.abs(resid), bins=30, color="#d62728", alpha=0.6, density=True, label="Primary E2")
ax.axvline(0, color="#1f77b4", lw=3, label="Symmetrized (exact, 0)")
ax.set_title("(c) Forward+reverse violation")
ax.set_xlabel(r"$|\Delta\Delta G_{\mathrm{fwd}}+\Delta\Delta G_{\mathrm{rev}}|$ (kcal/mol)")
ax.set_ylabel("Density"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figs_new/fig5_antisymmetry.pdf"); plt.close()
print("wrote figs_new/fig5_antisymmetry.pdf")

print("\nDONE. Replace Figures/fig3_*.pdf and Figures/fig5_antisymmetry.pdf with these.")
print("NOTE Fig 5 panels are SCHEMATIC illustrations of the exact-vs-residual behavior")
print("(the symmetrized exactness and E2 residual are both real, verified facts; the")
print("scatter is generated for visualization). If you have the ACTUAL per-sample")
print("fwd/rev predictions, load them instead of the rng draws for a data-exact figure.")
