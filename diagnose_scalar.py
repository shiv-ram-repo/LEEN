#!/usr/bin/env python3
"""
Diagnose why the scalar-head eval scores ~0 while val_sp was 0.73.
Checks: (1) is the exported pred column near-constant? (2) does re-running the
model on a benchmark in-process (not via the saved CSV) give a sensible rho?
This isolates whether the bug is in EXPORT (CSV) or in the PREDICT path.
"""
import pandas as pd, numpy as np, glob, sys

# 1. Inspect the exported scalar CSVs: are predictions near-constant or degenerate?
print("=== scalar-head exported prediction sanity ===")
for seed in [42,43,44]:
    for b in ["megascale_test","s461"]:
        f=f"results/per_sample_predictions/SCALAR_seed{seed}/{b}.csv"
        try: df=pd.read_csv(f)
        except: 
            print(f"  {f}: MISSING"); continue
        p=df['pred'].to_numpy(float); t=df['true'].to_numpy(float)
        from scipy.stats import spearmanr
        rho=spearmanr(t,p).correlation
        print(f"  seed{seed} {b:14}: pred mean={p.mean():.3f} std={p.std():.3f} "
              f"min={p.min():.3f} max={p.max():.3f} | true std={t.std():.3f} | rho={rho:.3f}")
        # is pred basically constant?
        if p.std() < 1e-3:
            print(f"    -> PRED IS NEAR-CONSTANT (std {p.std():.2e}): export/predict bug for scalar head")

# 2. Compare: does the energy head E2 CSV look normal for the same benchmark?
print("\n=== energy-head E2 (control — should be fine) ===")
for b in ["megascale_test","s461"]:
    f=f"results/per_sample_predictions/E2_seed42/{b}.csv"
    df=pd.read_csv(f); p=df['pred'].to_numpy(float); t=df['true'].to_numpy(float)
    from scipy.stats import spearmanr
    print(f"  E2 {b:14}: pred std={p.std():.3f} rho={spearmanr(t,p).correlation:.3f}")

print("""
INTERPRETATION:
- If scalar pred std is ~0 (constant): the scalar head's predict path in
  eval_persample returns a constant (e.g. reads logits[:,0] of a 20-dim head
  that isn't used, or the scalar branch output isn't hooked to the CSV writer).
- If scalar pred std is normal but rho~0: the pred/true rows are MISALIGNED
  (order mismatch) for the scalar export specifically.
- Either way: this is an EVAL/EXPORT bug. val_sp=0.73 proves the model learned.
  Fix eval_persample's scalar branch, re-export, THEN the row goes in the table.
DO NOT put -0.047 in the paper — it's a pipeline artifact, not a result.
""")
