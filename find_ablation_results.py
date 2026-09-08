#!/usr/bin/env python3
"""
Find ALL ablation prediction results across old + new directories and
compute Spearman per (config, benchmark, seed). Prints a table you can
paste back so we finalize tab:ablations from real numbers.

Searches these roots for per-sample CSVs:
  runs/leen_preds, results/per_sample_predictions, and any *preds* dir.

Recognizes ablation configs by directory-name substrings:
  SCALAR / scalar        -> scalar-head
  NOESMDROP / noesmdrop  -> no-ESM-dropout
  noOOD / no_ood / noood -> no-OOD
  E2 / leen_v2_drop      -> E2 baseline (energy head, full)
"""
import os, glob, re, sys
import numpy as np
try:
    import pandas as pd
    from scipy.stats import spearmanr
except ImportError:
    sys.exit("need pandas+scipy: pip install pandas scipy")

ROOTS = ["runs/leen_preds", "results/per_sample_predictions",
         "/backup/new_work/runs/leen_preds"]
# add any dir containing 'preds'
for d in glob.glob("**/*preds*", recursive=True):
    if os.path.isdir(d) and d not in ROOTS: ROOTS.append(d)

PRED=["pred","y_pred","prediction","ddg_pred","yhat","pred_ddg"]
TRUE=["true","y_true","target","ddg","label","y","exp_ddg","ddg_true"]
BENCH=["s669","s461","s783","s2648","s8754","ssym_direct","ssym_inverse",
       "s571","s4346","fireport_hf","fireprot_hf","megascale_test"]

def pick(cols,c):
    low={x.lower():x for x in cols}
    for k in c:
        if k in low: return low[k]
    return None

def classify(path):
    p=path.lower()
    if "scalar" in p: return "SCALAR"
    if "noesmdrop" in p or "no_esm" in p or "noesm" in p: return "NOESMDROP"
    if "noood" in p or "no_ood" in p or "noood" in p or "no-ood" in p: return "E2noOOD"
    if "leen_v2_drop" in p or re.search(r'\be2\b',p) or "/e2_" in p or "e2seed" in p: return "E2"
    return None

def seed_of(path):
    m=re.search(r'seed[_]?(\d+)', path.lower())
    return int(m.group(1)) if m else None

rows=[]
seen=set()
for root in ROOTS:
    for csv in glob.glob(f"{root}/**/*.csv", recursive=True):
        cfg=classify(csv); sd=seed_of(csv)
        if cfg is None or sd is None: continue
        bench=None
        for b in BENCH:
            if b in os.path.basename(csv).lower(): bench=b; break
        if bench is None: continue
        key=(cfg,bench,sd)
        if key in seen: continue
        try:
            df=pd.read_csv(csv)
        except Exception: continue
        pc=pick(df.columns,PRED); tc=pick(df.columns,TRUE)
        if pc is None or tc is None: continue
        y,yp=df[tc].to_numpy(float),df[pc].to_numpy(float)
        m=np.isfinite(y)&np.isfinite(yp); y,yp=y[m],yp[m]
        if len(y)<3: continue
        rho=spearmanr(y,yp).correlation
        rows.append(dict(config=cfg,benchmark=bench,seed=sd,spearman=round(float(rho),4),n=len(y),path=csv))
        seen.add(key)

if not rows:
    print("No ablation CSVs found. Roots searched:")
    for r in ROOTS: print("  ",r, "(exists)" if os.path.isdir(r) else "(missing)")
    sys.exit(0)

df=pd.DataFrame(rows).sort_values(["config","benchmark","seed"])
df.to_csv("ablation_found.csv",index=False)
print("=== per (config,benchmark,seed) ===")
print(df[["config","benchmark","seed","spearman","n"]].to_string(index=False))
print("\n=== seeds present per config ===")
print(df.groupby("config").seed.apply(lambda s: sorted(set(s))))
print("\n=== 3-seed (or available) MEAN per config x benchmark ===")
piv=df.groupby(["config","benchmark"]).spearman.agg(["mean","std","count"]).round(4)
print(piv.to_string())
print("\nSaved ablation_found.csv — paste the two summary blocks back.")
