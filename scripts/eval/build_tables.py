#!/usr/bin/env python3
"""Single source of truth for the LEEN revision. Reads per-sample predictions
(runs/leen_preds/<LABEL>_seed<N>/<benchmark>.csv) and regenerates EVERYTHING the reviewers asked for:

  * all four metrics (Spearman, Pearson, RMSE, MAE) per benchmark
  * mean +/- std across seeds
  * PROTEIN-CLUSTERED bootstrap 95% CIs (resample proteins, not mutations)
  * PAIRED bootstrap tests between configs on shared samples (e.g. E2 vs G2, V1 vs E2)
  * a coverage report flagging any missing config/seed/benchmark (kills the 8/9/11 inconsistency)

Every number in the manuscript should trace to this script's output.

  python build_leen_tables.py --preds_root runs/leen_preds --out_dir runs/leen_tables \
     --configs G1 G2 E1 E2 V1 --seeds 42 43 44
"""
from __future__ import annotations
import argparse, glob, itertools, json, os, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

BENCHMARKS = ["s669", "s461", "s783", "s2648", "s8754", "s571", "s4346",
              "ssym_direct", "ssym_inverse", "fireport_hf", "megascale_test"]


def metrics(pred, true):
    m = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[m], true[m]
    if len(p) < 3:
        return dict(spearman=np.nan, pearson=np.nan, rmse=np.nan, mae=np.nan, n=len(p))
    return dict(spearman=float(spearmanr(p, t).correlation),
                pearson=float(pearsonr(p, t)[0]),
                rmse=float(np.sqrt(np.mean((p - t) ** 2))),
                mae=float(np.mean(np.abs(p - t))), n=int(len(p)))


def protein_bootstrap(df, metric="spearman", n_boot=1000, seed=0):
    """Resample PROTEINS with replacement (cluster unit = protein_idx), recompute metric."""
    rng = np.random.default_rng(seed)
    groups = {pid: g[["pred", "true"]].values for pid, g in df.groupby("protein_idx")}
    pids = list(groups)
    if len(pids) < 3:
        return (np.nan, np.nan)
    fn = {"spearman": lambda a, b: spearmanr(a, b).correlation,
          "pearson": lambda a, b: pearsonr(a, b)[0],
          "rmse": lambda a, b: np.sqrt(np.mean((a - b) ** 2)),
          "mae": lambda a, b: np.mean(np.abs(a - b))}[metric]
    vals = []
    for _ in range(n_boot):
        samp = np.vstack([groups[pids[i]] for i in rng.integers(0, len(pids), len(pids))])
        try:
            v = fn(samp[:, 0], samp[:, 1])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    if len(vals) < 20:
        return (np.nan, np.nan)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def load(preds_root, cfg, seed, bench):
    p = os.path.join(preds_root, f"{cfg}_seed{seed}", f"{bench}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def paired_bootstrap(preds_root, a, b, seed, bench, metric="spearman", n_boot=1000):
    """Paired test: on the SAME proteins, is config a's metric > config b's? Returns P(a>b).
    Pre-groups proteins into numpy arrays ONCE, then indexes — ~100x faster than re-filtering."""
    da, db = load(preds_root, a, seed, bench), load(preds_root, b, seed, bench)
    if da is None or db is None:
        return np.nan
    m = da.merge(db, on=["protein_idx", "mut_index"], suffixes=("_a", "_b"))
    if len(m) < 10:
        return np.nan
    fn = (lambda x, y: spearmanr(x, y).correlation) if metric == "spearman" else (lambda x, y: pearsonr(x, y)[0])
    blocks = {pid: g[["pred_a", "true_a", "pred_b", "true_b"]].to_numpy()
              for pid, g in m.groupby("protein_idx")}
    pids = np.array(list(blocks))
    if len(pids) < 3:
        return np.nan
    rng = np.random.default_rng(0)
    wins = tot = 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(pids), len(pids))
        samp = np.vstack([blocks[pids[i]] for i in idx])
        try:
            va = fn(samp[:, 0], samp[:, 1]); vb = fn(samp[:, 2], samp[:, 3])
            if np.isfinite(va) and np.isfinite(vb):
                wins += (va > vb); tot += 1
        except Exception:
            pass
    return float(wins / tot) if tot else np.nan


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--preds_root", required=True); pa.add_argument("--out_dir", required=True)
    pa.add_argument("--configs", nargs="+", default=["G1", "G2", "E1", "E2", "V1"])
    pa.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    pa.add_argument("--pairs", nargs="+", default=["E2:G2", "E2:E1", "V1:E2", "E1:G2"],
                    help="paired significance comparisons a:b")
    args = pa.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    long_rows, coverage = [], []
    for cfg, bench in itertools.product(args.configs, BENCHMARKS):
        for seed in args.seeds:
            df = load(args.preds_root, cfg, seed, bench)
            if df is None or len(df) == 0:
                coverage.append({"config": cfg, "benchmark": bench, "seed": seed, "status": "MISSING"})
                continue
            mm = metrics(df["pred"].values, df["true"].values)
            mm.update(config=cfg, benchmark=bench, seed=seed)
            long_rows.append(mm)
            coverage.append({"config": cfg, "benchmark": bench, "seed": seed, "status": f"ok n={mm['n']}"})
    long = pd.DataFrame(long_rows)
    long.to_csv(os.path.join(args.out_dir, "per_seed_metrics.csv"), index=False)

    summ = []
    for cfg, bench in itertools.product(args.configs, BENCHMARKS):
        sub = long[(long.config == cfg) & (long.benchmark == bench)]
        if len(sub) == 0:
            continue
        row = {"config": cfg, "benchmark": bench, "n_seeds": len(sub)}
        for met in ("spearman", "pearson", "rmse", "mae"):
            row[f"{met}_mean"] = float(sub[met].mean())
            row[f"{met}_std"] = float(sub[met].std(ddof=1)) if len(sub) > 1 else 0.0
        d42 = load(args.preds_root, cfg, args.seeds[0], bench)
        if d42 is not None:
            lo, hi = protein_bootstrap(d42, "spearman")
            row["spearman_ci_lo"], row["spearman_ci_hi"] = lo, hi
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(args.out_dir, "summary_meanstd_ci.csv"), index=False)

    ptests = []
    for pair in args.pairs:
        a, b = pair.split(":")
        for bench in BENCHMARKS:
            probs = [paired_bootstrap(args.preds_root, a, b, s, bench) for s in args.seeds]
            probs = [p for p in probs if p == p]
            if probs:
                ptests.append({"comparison": f"{a} vs {b}", "benchmark": bench,
                               "P(a>b)": float(np.mean(probs)),
                               "significant": bool(np.mean(probs) > 0.95 or np.mean(probs) < 0.05)})
    pd.DataFrame(ptests).to_csv(os.path.join(args.out_dir, "paired_tests.csv"), index=False)

    cov = pd.DataFrame(coverage)
    cov.to_csv(os.path.join(args.out_dir, "coverage.csv"), index=False)
    missing = cov[cov.status == "MISSING"]

    print("\n=== SPEARMAN (mean ± std across seeds) ===")
    piv = summ.pivot(index="config", columns="benchmark", values="spearman_mean")
    piv = piv.reindex(index=args.configs, columns=[b for b in BENCHMARKS if b in piv.columns])
    print(piv.round(3).to_string())
    print(f"\nfull tables -> {args.out_dir}/  (per_seed_metrics, summary_meanstd_ci, paired_tests, coverage)")
    if len(missing):
        print(f"\n⚠ {len(missing)} MISSING config/seed/benchmark cells — see coverage.csv:")
        print(missing.groupby("config").size().to_string())


if __name__ == "__main__":
    main()
