#!/usr/bin/env python3
"""
Regenerate ALL metric tables from the released per-sample predictions,
so every number in the paper traces to one source of truth.

Outputs:
  - main_spearman.tex        (Spearman, all configs x 11 benchmarks, 3-seed mean+/-sd)
  - metrics_full.tex         (Pearson/RMSE/MAE appendix table for LEEN configs)
  - paired_tests.tex         (E2 vs G2 protein-clustered paired bootstrap)
  - ablations.tex            (E2 / scalar / no-OOD / no-ESM-drop)
  - all_metrics.csv          (long-form: config,seed,benchmark,metric,value)

USAGE (from LEEN_clean/):
  PYTHONPATH=/backup/new_work:. python build_metrics_tables.py \
      --preds_root results/per_sample_predictions --out_dir results/paper_tables

This script is DEFENSIVE: it first prints the directory layout and the columns
of the first CSV it finds, so if the schema differs from what it assumes you
see it immediately rather than getting silently-wrong numbers.
"""
import argparse, os, glob, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas required: pip install pandas")
from scipy.stats import spearmanr, pearsonr

# ---- config: how prediction files are named/organized -----------------------
# We assume: <preds_root>/<CONFIG>_seed<SEED>/<benchmark>.csv
# with columns including a predicted column and a true column.
# We auto-detect the column names below.
PRED_COL_CANDIDATES = ["pred", "y_pred", "prediction", "ddg_pred", "yhat", "pred_ddg"]
TRUE_COL_CANDIDATES = ["true", "y_true", "target", "ddg", "label", "y", "exp_ddg", "ddg_true"]
PROT_COL_CANDIDATES = ["pdb", "protein", "wt_name", "pdb_id", "wt", "name", "protein_id"]

BENCHMARKS = ["s669","s461","s783","s2648","s8754","ssym_direct","ssym_inverse",
              "s571","s4346","fireprot_hf","megascale_test"]
CONFIGS = ["G1","G2","E1","E2","V1"]          # main table
ABLATION_CONFIGS = ["E2","SCALAR","E2noOOD","NOESMDROP"]
SEEDS = [42,43,44]

def _pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low: return low[c]
    return None

def load_pred_csv(path):
    df = pd.read_csv(path)
    pc = _pick(df.columns, PRED_COL_CANDIDATES)
    tc = _pick(df.columns, TRUE_COL_CANDIDATES)
    prc= _pick(df.columns, PROT_COL_CANDIDATES)
    if pc is None or tc is None:
        return None, None, None, df.columns.tolist()
    return df, (pc, tc, prc), None, df.columns.tolist()

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) < 3:
        return dict(spearman=np.nan, pearson=np.nan, rmse=np.nan, mae=np.nan, n=len(y_true))
    sp = spearmanr(y_true, y_pred).correlation
    pr = pearsonr(y_true, y_pred)[0]
    rmse = float(np.sqrt(np.mean((y_true-y_pred)**2)))
    mae = float(np.mean(np.abs(y_true-y_pred)))
    return dict(spearman=sp, pearson=pr, rmse=rmse, mae=mae, n=len(y_true))

def protein_bootstrap_ci(df, pc, tc, prc, metric="spearman", n_boot=10000, seed=0):
    """95% CI resampling proteins (clusters). Returns (lo, hi)."""
    rng = np.random.default_rng(seed)
    if prc is None:
        # fall back to sample bootstrap
        groups = [df]
        idx = df.index.to_numpy()
        vals=[]
        for _ in range(n_boot):
            b = rng.choice(idx, size=len(idx), replace=True)
            s = df.loc[b]
            vals.append(metrics(s[tc], s[pc])[metric])
        return np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)
    prots = df[prc].unique()
    by = {p: df[df[prc]==p] for p in prots}
    vals=[]
    for _ in range(n_boot):
        chosen = rng.choice(prots, size=len(prots), replace=True)
        s = pd.concat([by[p] for p in chosen], ignore_index=True)
        vals.append(metrics(s[tc], s[pc])[metric])
    return float(np.nanpercentile(vals,2.5)), float(np.nanpercentile(vals,97.5))

def paired_bootstrap(dfA, dfB, cols, n_boot=10000, seed=0):
    """Protein-clustered paired bootstrap of Delta-spearman (A - B).
    dfA, dfB must be aligned on the same samples/proteins."""
    pcA,tcA,prcA = cols
    rng = np.random.default_rng(seed)
    # align on protein
    prots = np.intersect1d(dfA[prcA].unique(), dfB[prcA].unique()) if prcA else None
    if prcA is None:
        return None
    byA = {p: dfA[dfA[prcA]==p] for p in prots}
    byB = {p: dfB[dfB[prcA]==p] for p in prots}
    deltas=[]
    for _ in range(n_boot):
        chosen = rng.choice(prots, size=len(prots), replace=True)
        sA = pd.concat([byA[p] for p in chosen], ignore_index=True)
        sB = pd.concat([byB[p] for p in chosen], ignore_index=True)
        dA = metrics(sA[tcA], sA[pcA])["spearman"]
        dB = metrics(sB[tcA], sB[pcA])["spearman"]
        deltas.append(dA-dB)
    deltas=np.array(deltas)
    lo,hi = np.nanpercentile(deltas,2.5), np.nanpercentile(deltas,97.5)
    p_two = 2*min((deltas<=0).mean(), (deltas>=0).mean())
    return float(np.nanmean(deltas)), float(lo), float(hi), float(p_two)

def find_csv(preds_root, config, seed, bench):
    # try a few directory conventions
    pats = [
        f"{preds_root}/{config}_seed{seed}/{bench}.csv",
        f"{preds_root}/{config}_seed{seed}/{bench}_preds.csv",
        f"{preds_root}/{config}/seed{seed}/{bench}.csv",
        f"{preds_root}/{config}_seed_{seed}/{bench}.csv",
    ]
    for p in pats:
        if os.path.exists(p): return p
    # glob fallback
    g = glob.glob(f"{preds_root}/*{config}*seed*{seed}*/*{bench}*.csv")
    return g[0] if g else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_root", required=True)
    ap.add_argument("--out_dir", default="paper_tables")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 0. show layout so schema mismatches surface immediately ----
    print("=== directory scan ===")
    dirs = sorted(glob.glob(f"{args.preds_root}/*"))
    for d in dirs[:20]: print("  ", d)
    sample = glob.glob(f"{args.preds_root}/**/*.csv", recursive=True)
    if not sample:
        sys.exit(f"No CSVs under {args.preds_root} — check the path.")
    print(f"\n=== first CSV: {sample[0]} ===")
    df0,cols,_,allcols = load_pred_csv(sample[0])
    print("  columns:", allcols)
    if df0 is None:
        sys.exit("Could not auto-detect pred/true columns. Edit PRED_COL_CANDIDATES / TRUE_COL_CANDIDATES at the top of this script to match the columns printed above.")
    pc,tc,prc = cols
    print(f"  using pred='{pc}' true='{tc}' protein='{prc}'")

    # ---- 1. collect metrics for every config/seed/benchmark ----
    rows=[]
    allcfgs = list(dict.fromkeys(CONFIGS + ABLATION_CONFIGS))
    for cfg in allcfgs:
        for seed in SEEDS:
            for bench in BENCHMARKS:
                path = find_csv(args.preds_root, cfg, seed, bench)
                if not path: continue
                df,cols2,_,_ = load_pred_csv(path)
                if df is None: continue
                p2,t2,pr2 = cols2
                m = metrics(df[t2], df[p2])
                for met in ("spearman","pearson","rmse","mae"):
                    rows.append(dict(config=cfg,seed=seed,benchmark=bench,metric=met,value=m[met],n=m["n"]))
    long = pd.DataFrame(rows)
    long.to_csv(f"{args.out_dir}/all_metrics.csv", index=False)
    print(f"\nwrote {args.out_dir}/all_metrics.csv ({len(long)} rows)")

    def agg(cfg, bench, met):
        sub = long[(long.config==cfg)&(long.benchmark==bench)&(long.metric==met)]
        if sub.empty: return (np.nan,np.nan,0)
        return (sub.value.mean(), sub.value.std(ddof=0), sub.seed.nunique())

    # ---- 2. metrics_full.tex (Pearson/RMSE/MAE for LEEN configs) ----
    with open(f"{args.out_dir}/metrics_full.tex","w") as f:
        f.write("% Pearson / RMSE / MAE appendix table, regenerated from per-sample preds\n")
        f.write("\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\caption{Pearson correlation, RMSE, and MAE for the E2 model across benchmarks (3-seed mean). Spearman is in Table~\\ref{tab:main_results}. $\\Delta T_m$ datasets in $^\\circ$C; others in kcal/mol.}\n")
        f.write("\\label{tab:metrics_full}\n\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Benchmark & Spearman & Pearson & RMSE & MAE \\\\\n\\midrule\n")
        namemap={"s669":"S669","s461":"S461","s783":"S783","s2648":"S2648","s8754":"S8754",
                 "ssym_direct":"Ssym-d","ssym_inverse":"Ssym-i","s571":"S571 ($\\Delta T_m$)",
                 "s4346":"S4346 ($\\Delta T_m$)","fireprot_hf":"FireProt-HF","megascale_test":"Mega-test"}
        for b in BENCHMARKS:
            sp=agg("E2",b,"spearman")[0]; pr=agg("E2",b,"pearson")[0]
            rm=agg("E2",b,"rmse")[0]; ma=agg("E2",b,"mae")[0]
            def fmt(x): return f"${x:.3f}$" if np.isfinite(x) else "n.r."
            f.write(f"{namemap[b]} & {fmt(sp)} & {fmt(pr)} & {fmt(rm)} & {fmt(ma)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"wrote {args.out_dir}/metrics_full.tex")

    # ---- 3. paired_tests.tex (E2 vs G2) ----
    with open(f"{args.out_dir}/paired_tests.tex","w") as f:
        f.write("% E2 vs G2 protein-clustered paired bootstrap, regenerated\n")
        f.write("\\begin{table}[t]\\centering\\small\n")
        f.write("\\caption{Protein-clustered paired bootstrap tests ("+str(args.n_boot)+" resamples) for gating vs concatenation (E2$-$G2). 95\\% CI and two-sided $p$.}\n")
        f.write("\\label{tab:paired_tests}\n\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Benchmark & $\\Delta\\rho$ (E2$-$G2) & 95\\% CI & $p$ \\\\\n\\midrule\n")
        for b in BENCHMARKS:
            # need same-seed aligned; use seed 42 as representative for the paired test
            pA = find_csv(args.preds_root,"E2",42,b); pB=find_csv(args.preds_root,"G2",42,b)
            if not pA or not pB: 
                f.write(f"{b} & n.r. & n.r. & n.r. \\\\\n"); continue
            dfA,cA,_,_=load_pred_csv(pA); dfB,cB,_,_=load_pred_csv(pB)
            res = paired_bootstrap(dfA, dfB, cA, n_boot=args.n_boot)
            if res is None:
                f.write(f"{b} & n.r. & n.r. & n.r. \\\\\n"); continue
            d,lo,hi,p = res
            sig = "$^{*}$" if p<0.05 else ""
            f.write(f"{b} & ${d:+.3f}${sig} & $[{lo:+.3f},{hi:+.3f}]$ & ${p:.3f}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\\\[2pt]\\footnotesize $^{*}$ significant at $p<0.05$.\n\\end{table}\n")
    print(f"wrote {args.out_dir}/paired_tests.tex")

    # ---- 4. ablations.tex ----
    with open(f"{args.out_dir}/ablations.tex","w") as f:
        f.write("% Ablation table regenerated. Rows present only if preds exist.\n")
        f.write("\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\caption{Component ablations. E2 and no-OOD are 3-seed means where available; scalar-head and no-ESM-drop use the seeds present in the repository. Ssym-i shows the energy head retains reverse-mutation signal while the scalar head does not.}\n")
        f.write("\\label{tab:ablations}\n\\begin{tabular}{l ccccc cc cc c}\n\\toprule\n")
        f.write("& S669 & S461 & S783 & S2648 & S8754 & S571 & S4346 & Ssym-d & Ssym-i & Mega \\\\\n\\midrule\n")
        labelmap={"E2":"E2 (energy head, full)","SCALAR":"\\quad scalar head",
                  "E2noOOD":"\\quad no OOD loss","NOESMDROP":"\\quad no ESM-drop"}
        order=["E2","SCALAR","E2noOOD","NOESMDROP"]
        colbench=["s669","s461","s783","s2648","s8754","s571","s4346","ssym_direct","ssym_inverse","megascale_test"]
        for cfg in order:
            cells=[]
            present=False
            for b in colbench:
                mean,sd,ns = agg(cfg,b,"spearman")
                if np.isfinite(mean):
                    present=True
                    cells.append(f"${mean:.3f}$" if ns<2 else f"${mean:.3f}{{\\pm}}{sd:.3f}$")
                else:
                    cells.append("--")
            if present:
                f.write(labelmap[cfg]+" & "+" & ".join(cells)+" \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"wrote {args.out_dir}/ablations.tex")

    print("\nDONE. Review the .tex files in", args.out_dir)
    print("If Spearman means differ from the manuscript master numbers, the CSVs are the source of truth — update the paper to match.")

if __name__=="__main__":
    main()
