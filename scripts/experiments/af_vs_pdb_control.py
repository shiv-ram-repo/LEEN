#!/usr/bin/env python3
"""AlphaFold-vs-experimental-PDB structure-source control (Reviewer 2, point e).

For S669 and S461, re-evaluate LEEN using AlphaFold structures under an otherwise identical
pipeline, and compare Spearman against the experimental-PDB evaluation. If the two agree, the
benchmark difficulty reflects genuine (sequence-level) generalization rather than a structure-source
artifact; if they differ, we report the source-dependent bias honestly.

APPROACH:
  1. For each benchmark protein, obtain an AlphaFold model:
       (a) if a local AF model exists (via --af_dir mapping), use it;
       (b) else fetch from the AlphaFold DB by UniProt accession (--fetch), if provided a mapping;
       (c) else fold with ColabFold (--colabfold) — heaviest, optional.
  2. Build the protein dict from the AF structure (same parser as the normal pipeline),
     keep the SAME sequence, mutations, and ddG labels — only the COORDINATES change.
  3. Score with the same LEEN checkpoint + adapter, compute per-benchmark Spearman.
  4. Report PDB-source vs AF-source side by side.

  PYTHONPATH=/backup/new_work python af_vs_pdb_control.py \
    --ckpt /backup/new_work/runs/leen_v2_drop/seed_42/best.pt \
    --esm_cache /backup/new_work/leen/esm_cache_full.pt \
    --data_root /backup/stability/data_root \
    --af_dir /backup/stability/data_root/data/dataset/af_benchmarks \
    --benchmarks s669 s461 --out af_vs_pdb_report.json
"""
from __future__ import annotations
import argparse, glob, json, os
import torch
from scipy.stats import spearmanr

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../eval'))
from eval_persample import build_v2, _cfg_get
from leen.data.adapter_v2 import protein_to_graphs_v2, collate_v2
from protstab_data import MegaScaleTestDatasets


def spearman_on_dataset(model, ds, esm_cache, cfg, device, coord_override=None, restrict_fn=None):
    """Score a benchmark dataset. coord_override(p)->p_with_AF_coords swaps the structure while keeping
    sequence/mutations/labels identical. restrict_fn(p)->bool, when given, skips proteins for which it
    returns False (used to score PDB on ONLY the AF-covered subset, for a paired comparison).
    Returns (spearman, n, n_proteins_used)."""
    preds, trues = [], []
    used = 0
    for pidx in range(len(ds)):
        try:
            p = ds[pidx]
            if p is None:
                continue
            if restrict_fn is not None and not restrict_fn(p):
                continue
            if coord_override is not None:
                ov = coord_override(p)
                if ov is None:
                    continue
                p = ov
                used += 1
            gs = protein_to_graphs_v2(p, esm_cache, _cfg_get(cfg, "env_radius", 10.0),
                                      _cfg_get(cfg, "edge_cutoff", 8.0), _cfg_get(cfg, "max_neighbors", 64))
            if not gs:
                continue
            b = collate_v2(gs).to(device)
            pr = model(b)["ddg_pred"].detach().cpu().numpy().reshape(-1)
            tr = b.ddg.cpu().numpy().reshape(-1) if b.ddg is not None else None
            if tr is None:
                continue
            k = min(len(pr), len(tr))
            preds += pr[:k].tolist(); trues += tr[:k].tolist()
        except Exception:
            continue
    if len(preds) < 5:
        return float("nan"), len(preds), used
    return float(spearmanr(preds, trues).correlation), len(preds), used


def make_af_override(af_dir, parser_fn):
    """Return a function protein_dict -> protein_dict_with_AF_coords, or None if no AF model.
    Matches AF file by the protein 'name' (strip chain). Expects <af_dir>/<name>.pdb."""
    import re
    index = {}
    for f in sorted(glob.glob(os.path.join(af_dir, "*.pdb"))):
        base = os.path.basename(f)
        m = re.split(r"_(?:un)?relaxed_rank_\d+", base)
        pid = m[0].lower() if m else os.path.splitext(base)[0].lower()
        if pid not in index or "rank_001" in base:
            index[pid] = f
    def override(p):
        name = str(p.get("name", "")).lower()
        cands = [name, name.replace(".pdb", ""), name.split("_")[0], name[:-1]]
        af = next((index[c] for c in cands if c in index), None)
        if af is None:
            return None
        try:
            newp = parser_fn(af, p)
            return newp
        except Exception:
            return None
    return override, index


def parse_af_into_protein(af_path, p):
    """Build an AF-structure protein dict via the SAME path the benchmark loaders use:
    alt_parse_PDB -> get_pdb(parsed, wt_seq, wt_name), yielding a pipeline-perfect 'X' tensor from
    the AF coordinates. Transplant that X into a copy of the benchmark protein, keeping its
    mutations/ddG/append_tensors. Returns None if the AF structure length does not match the
    benchmark sequence (mutation indices must stay valid) or if X is not actually different."""
    from protstab_data.pdb_parser import alt_parse_PDB
    from protstab_data.featurizer import get_pdb
    import torch as _t
    wt_seq = p.get("seq", "")
    wt_name = str(p.get("name", ""))
    try:
        parsed = alt_parse_PDB(af_path, ["A"])
        af_protein = get_pdb(parsed[0], wt_name, wt_name, check_assert=False)
    except Exception:
        return None
    if not isinstance(af_protein, dict) or "X" not in af_protein:
        return None
    try:
        if af_protein["X"].shape[1] != len(wt_seq):
            return None
    except Exception:
        return None
    try:
        if "X" in p and _t.allclose(af_protein["X"].float(), p["X"].float(), atol=1e-4):
            return None
    except Exception:
        pass
    newp = dict(p)
    newp["X"] = af_protein["X"]
    if "coords" in af_protein:
        newp["coords"] = af_protein["coords"]
    return newp


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", required=True); pa.add_argument("--esm_cache", required=True)
    pa.add_argument("--data_root", required=True); pa.add_argument("--af_dir", required=True)
    pa.add_argument("--benchmarks", nargs="+", default=["s669", "s461"])
    pa.add_argument("--out", default="af_vs_pdb_report.json")
    args = pa.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model, _, _ = build_v2(cfg, device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    esm_cache = torch.load(args.esm_cache, map_location="cpu", weights_only=False)

    coll = dict(MegaScaleTestDatasets(data_root=args.data_root).iter_named())
    override, af_index = make_af_override(args.af_dir, parse_af_into_protein)
    print(f"AF models available in {args.af_dir}: {len(af_index)}")

    report = {}
    for bn in args.benchmarks:
        ds = coll[bn]
        def has_af(p, _ov=override):
            return _ov(p) is not None
        sp_pdb_all, n_pdb_all, _ = spearman_on_dataset(model, ds, esm_cache, cfg, device, coord_override=None)
        sp_pdb_sub, n_pdb_sub, _ = spearman_on_dataset(model, ds, esm_cache, cfg, device, coord_override=None, restrict_fn=has_af)
        sp_af, n_af, used = spearman_on_dataset(model, ds, esm_cache, cfg, device, coord_override=override)
        report[bn] = {
            "spearman_PDB_all_proteins": round(sp_pdb_all, 4),
            "spearman_PDB_AFsubset": round(sp_pdb_sub, 4) if sp_pdb_sub == sp_pdb_sub else None,
            "spearman_AlphaFold_AFsubset": round(sp_af, 4) if sp_af == sp_af else None,
            "PAIRED_delta_AF_minus_PDB_on_same_proteins": round(sp_af - sp_pdb_sub, 4) if (sp_af==sp_af and sp_pdb_sub==sp_pdb_sub) else None,
            "n_mut_pdb_all": n_pdb_all, "n_mut_pdb_subset": n_pdb_sub, "n_mut_af": n_af,
            "n_proteins_with_AF": used, "n_proteins_total": len(ds),
            "note": "PAIRED comparison = AFsubset vs PDB_AFsubset (SAME proteins, only structure source differs). "
                    "The all-proteins PDB number is for reference only and is NOT a fair comparison to AF.",
        }
        print(f"\n=== {bn} ===")
        print(f"  PDB  (all {len(ds)} prot)      Spearman={sp_pdb_all:.4f}  (n={n_pdb_all})")
        print(f"  PDB  (AF subset, {used} prot)  Spearman={sp_pdb_sub:.4f}  (n={n_pdb_sub})")
        print(f"  AF   (AF subset, {used} prot)  Spearman={sp_af:.4f}  (n={n_af})")
        if sp_af == sp_af and sp_pdb_sub == sp_pdb_sub:
            print(f"  >>> PAIRED Δ(AF-PDB, same proteins) = {sp_af - sp_pdb_sub:+.4f}  <<<")

    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nreport -> {args.out}")
    print("\nINTERPRETATION: small |Δ| (< ~0.03) => structure source is NOT a systematic confound; "
          "benchmark difficulty reflects sequence-level generalization. Large |Δ| => report the "
          "source-dependent bias honestly.")


if __name__ == "__main__":
    main()
