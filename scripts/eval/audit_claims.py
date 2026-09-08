#!/usr/bin/env python3
"""Mechanical audit of the paper's numerical claims against the single source of truth
(runs/leen_preds per-sample CSVs). Answers Reviewer 2's demand to regenerate every number
from one set of raw predictions and flag contradictions.

Run build_leen_tables.py FIRST to produce per_seed_metrics.csv, then point this at it.

  python audit_paper_claims.py --metrics_csv /backup/new_work/runs/leen_tables_full/per_seed_metrics.csv
"""
import argparse, itertools
import numpy as np, pandas as pd

ALL_BENCHMARKS = ["s669","s461","s783","s2648","s8754","s571","s4346",
                  "ssym_direct","ssym_inverse","fireport_hf","megascale_test"]

def mean_std(df, cfg, bench, metric="spearman"):
    d = df[(df.config==cfg) & (df.benchmark==bench)][metric]
    return (float(d.mean()), float(d.std())) if len(d) else (np.nan, np.nan)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--metrics_csv", required=True)
    args = pa.parse_args()
    df = pd.read_csv(args.metrics_csv)
    configs = sorted(df.config.unique())
    benches = [b for b in ALL_BENCHMARKS if b in set(df.benchmark)]

    print("="*70)
    print("MASTER SPEARMAN TABLE (mean±std across seeds) — SINGLE SOURCE OF TRUTH")
    print("="*70)
    hdr = f"{'benchmark':16s}" + "".join(f"{c:>14s}" for c in configs)
    print(hdr)
    for b in benches:
        row = f"{b:16s}"
        for c in configs:
            m,s = mean_std(df,c,b)
            row += f"  {m:.3f}±{s:.3f}" if m==m else f"{'--':>14s}"
        print(row)

    print("\n" + "="*70)
    print("CLAIM CHECKS (flag = paper statement contradicts the data)")
    print("="*70)

    print("\n[1] E2->V1 deltas (paper Section 5.3 claims +0.018/+0.012/+0.024):")
    for b in ["s669","s461","s571"]:
        e2,_ = mean_std(df,"E2",b); v1,_ = mean_std(df,"V1",b)
        if e2==e2 and v1==v1:
            print(f"    {b:12s} E2={e2:.3f} V1={v1:.3f}  ACTUAL Δ(V1-E2)={v1-e2:+.3f}")

    print("\n[2] GVP(V1) vs EGNN(E2) across external benchmarks (claim: GVP consistently better):")
    externals = [b for b in benches if b!="megascale_test"]
    v1_wins=e2_wins=0
    for b in externals:
        e2,_ = mean_std(df,"E2",b); v1,_ = mean_std(df,"V1",b)
        if e2==e2 and v1==v1:
            w = "V1" if v1>e2 else "E2"
            v1_wins += v1>e2; e2_wins += e2>=v1
            print(f"    {b:12s} E2={e2:.3f} V1={v1:.3f}  winner={w}")
    print(f"    => V1 wins {v1_wins}/{len(externals)}, E2 wins {e2_wins}/{len(externals)}  "
          f"=> {'COMPARABLE (claim UNSUPPORTED)' if abs(v1_wins-e2_wins)<=1 else 'one is clearly better'}")

    print(f"\n[3] Benchmark count: data has {len(benches)} benchmarks: {benches}")
    print(f"    => state ONE consistent number everywhere (paper variously says 11/10/9/7).")

    print("\n[4] Std-dev check (paper claims all σ < 0.015):")
    viol=[]
    for c,b in itertools.product(configs, benches):
        _,s = mean_std(df,c,b)
        if s==s and s>=0.015: viol.append((c,b,s))
    if viol:
        print(f"    VIOLATIONS ({len(viol)} config×benchmark with σ≥0.015):")
        for c,b,s in sorted(viol,key=lambda x:-x[2])[:10]:
            print(f"      {c} {b}: σ={s:.3f}")
        print("    => correct the 'all σ<0.015' claim; report actual σ.")
    else:
        print("    all σ < 0.015 — claim holds.")

    print("\n[5] Gating(E2) vs Concat(G2) across benchmarks (claim: gating better on 7/9):")
    if "G2" in configs:
        gw=0; tot=0
        for b in benches:
            e2,_=mean_std(df,"E2",b); g2,_=mean_std(df,"G2",b)
            if e2==e2 and g2==g2:
                tot+=1; gw+= e2>g2
        print(f"    => E2>G2 on {gw}/{tot} benchmarks (mean-only; use paired_tests.csv for significance).")

    print("\n[6] Coverage completeness (benchmarks missing any config/seed):")
    for b in benches:
        for c in configs:
            n = len(df[(df.config==c)&(df.benchmark==b)])
            if n < 3:
                print(f"    {c} {b}: only {n} seeds (need 3 for mean±std)")
    print("\nDONE. Use the master table above as the SOLE source; fix every flagged claim in the paper.")

if __name__ == "__main__":
    main()
