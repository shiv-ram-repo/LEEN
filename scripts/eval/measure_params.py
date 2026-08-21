#!/usr/bin/env python3
"""Honest parameter + compute accounting table (answers Reviewers 1 & 3).

Reports, for LEEN and each baseline, a CONSISTENT accounting:
  - task-specific TRAINABLE params
  - TOTAL loaded params (incl. frozen ESM-2 backbone dependency)
  - FROZEN dependency params
  - model file size on disk (trainable checkpoint)
  - peak GPU memory at inference
  - ESM-2 preprocessing time (first eval of a new protein; amortizable)
  - cached mutation-inference throughput (mut/s, ESM reused)

This makes the accounting fair: LEEN's frozen 650M ESM-2 dependency is shown explicitly, and the
"trainable-only" and "total" conventions are both reported so no single convention is cherry-picked.

  PYTHONPATH=/backup/new_work python measure_param_efficiency.py \
    --ckpt /backup/new_work/runs/leen_v2_drop/seed_42/best.pt \
    --esm_cache /backup/new_work/leen/esm_cache_full.pt \
    --data_root /backup/stability/data_root
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))
from eval_persample import build_v2, _cfg_get
from leen.data.adapter_v2 import protein_to_graphs_v2, collate_v2
from protstab_data import MegaScaleTestDatasets

ESM2_650M_PARAMS = 650_000_000


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train, total


def file_size_mb(path):
    return os.path.getsize(path) / 1e6


@torch.no_grad()
def measure_inference(model, ckpt_cfg, esm_cache, data_root, device, n_proteins=20):
    """Cached-inference throughput (ESM reused) + peak memory."""
    coll = MegaScaleTestDatasets(data_root=data_root)
    ds = dict(coll.iter_named())["s669"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    n_mut, t0 = 0, time.time()
    for pidx in range(min(len(ds), n_proteins)):
        try:
            p = ds[pidx]
            if p is None:
                continue
            gs = protein_to_graphs_v2(p, esm_cache, _cfg_get(ckpt_cfg, "env_radius", 10.0),
                                      _cfg_get(ckpt_cfg, "edge_cutoff", 8.0), _cfg_get(ckpt_cfg, "max_neighbors", 64))
            if not gs:
                continue
            b = collate_v2(gs).to(device)
            _ = model(b)["ddg_pred"]
            n_mut += b.ddg.numel() if b.ddg is not None else len(gs)
        except Exception:
            continue
    dt = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else float("nan")
    return (n_mut / dt if dt > 0 else float("nan")), peak_mb, n_mut


@torch.no_grad()
def measure_esm_preprocess(data_root, device, n_proteins=10):
    """Time to run ESM-2 650M on new proteins (the preprocessing cost LEEN amortizes via caching)."""
    try:
        import esm
    except Exception as e:
        return None, f"esm not importable: {e}"
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()
    ds = dict(MegaScaleTestDatasets(data_root=data_root).iter_named())["s669"]
    seqs = []
    for i in range(min(len(ds), n_proteins)):
        p = ds[i]
        if p is not None and "seq" in p:
            seqs.append(p["seq"][:1022])
    if not seqs:
        return None, "no seqs"
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for s in seqs:
        _, _, toks = bc([("x", s)]); toks = toks.to(device)
        _ = model(toks, repr_layers=[model.num_layers])["representations"][model.num_layers]
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return dt / len(seqs), f"{len(seqs)} proteins, mean {dt/len(seqs):.3f}s/protein"


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", required=True); pa.add_argument("--esm_cache", required=True)
    pa.add_argument("--data_root", required=True); pa.add_argument("--out", default="param_efficiency.json")
    args = pa.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model, _, _ = build_v2(cfg, device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    train_p, total_p = count_params(model)
    esm_cache = torch.load(args.esm_cache, map_location="cpu", weights_only=False)

    thr, peak_mb, n_mut = measure_inference(model, cfg, esm_cache, args.data_root, device)
    esm_t, esm_note = measure_esm_preprocess(args.data_root, device)

    report = {
        "LEEN (this work)": {
            "trainable_params": train_p,
            "trainable_params_M": round(train_p / 1e6, 2),
            "frozen_ESM2_dependency_params": ESM2_650M_PARAMS,
            "total_loaded_params": train_p + ESM2_650M_PARAMS,
            "total_loaded_params_M": round((train_p + ESM2_650M_PARAMS) / 1e6, 1),
            "trainable_checkpoint_size_MB": round(file_size_mb(args.ckpt), 2),
            "peak_gpu_memory_MB_inference": round(peak_mb, 1) if peak_mb == peak_mb else None,
            "cached_inference_throughput_mut_per_s": round(thr, 1) if thr == thr else None,
            "esm_preprocess_s_per_protein": round(esm_t, 3) if esm_t else None,
            "esm_preprocess_note": esm_note,
            "n_mutations_timed": n_mut,
        },
        "_accounting_notes": {
            "convention": "trainable = task-specific params optimized; total = trainable + frozen ESM-2 dependency",
            "esm2_650M_params": ESM2_650M_PARAMS,
            "fair_comparison": "For a fair comparison, report BOTH trainable-only and total for every method; "
                               "ESM-2 zero-shot has ~0 trainable and 650M total; LEEN has ~3M trainable and ~653M total.",
            "TODO_baselines": "Fill SPURS / ThermoMPNN / ESM-2 zero-shot trainable+total from their code/papers "
                              "(SPURS and ThermoMPNN also build on frozen backbones — count consistently).",
        },
    }
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
