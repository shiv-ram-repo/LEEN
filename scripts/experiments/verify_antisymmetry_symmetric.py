#!/usr/bin/env python3
"""Prove EXACT independent-input anti-symmetry for the masked model.

The reviewers' real test: build FORWARD (A->V) from the WT sequence/structure, and REVERSE (V->A)
from the MUTANT sequence/structure — as INDEPENDENT inputs — then check ΔΔĜ_fwd + ΔΔĜ_rev.

With dual masking, WT and mutant inputs differ only at the masked site -> identical context ->
the sum is 0 to floating-point precision. (ACDC-NN only drives this ~0 via a penalty; we get it exact.)

Works on ANY checkpoint (even the un-retrained E2) because anti-symmetry is architectural, not learned.
Accuracy will be poor on an un-retrained model — that's fine; this checks the PROPERTY, not accuracy.

  PYTHONPATH=/backup/new_work python verify_antisymmetry.py \
    --ckpt /backup/new_work/runs/leen_v2_drop/seed_42/best.pt \
    --sym_esm_cache masked_esm_cache_bench.pt \
    --data_root /backup/stability/data_root --benchmark ssym_direct --n 200
"""
from __future__ import annotations
import argparse, copy, importlib.util, os, sys
import numpy as np, torch
from protstab_data import MegaScaleTestDatasets

here = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("ev", os.path.join(here, "eval_leen_persample.py"))
ev = importlib.util.module_from_spec(spec); sys.modules["ev"] = ev; spec.loader.exec_module(ev)
from leen.data.adapter_v2_symmetric import protein_to_graphs_v2_symmetric
from leen.data.adapter_v2 import collate_v2


def apply_mutation_to_S(protein):
    """Return a copy of the protein with S mutated wt->mut at each mut site, and the mutation swapped,
    so it represents the REVERSE mutation from the MUTANT scaffold (independent input)."""
    import copy
    p = copy.deepcopy(protein)
    S_t = p["S"]
    orig_shape = S_t.shape
    S = S_t.reshape(-1).clone()
    at = p["append_tensors"].clone() if torch.is_tensor(p["append_tensors"]) else torch.as_tensor(np.asarray(p["append_tensors"]))
    for mi, m in enumerate(p["mut_ids"]):
        pos = int(m)
        wt = int(at[mi, :21].argmax()); mt = int(at[mi, 21:].argmax())
        if 0 <= pos < S.numel():
            S[pos] = mt
        at[mi, :21] = 0; at[mi, 21:] = 0
        at[mi, mt] = 1; at[mi, 21 + wt] = 1
    p["S"] = S.reshape(orig_shape)
    p["append_tensors"] = at
    return p


@torch.no_grad()
def score(model, protein, masked_cache, device):
    gs = protein_to_graphs_v2_symmetric(protein, esm_cache=None, sym_esm_cache=masked_cache)
    if not gs:
        return None
    b = collate_v2(gs).to(device)
    return model(b)["ddg_pred"].cpu().numpy().reshape(-1)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", required=True); pa.add_argument("--sym_esm_cache", required=True)
    pa.add_argument("--data_root", required=True); pa.add_argument("--benchmark", default="ssym_direct")
    pa.add_argument("--n", type=int, default=200)
    args = pa.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, _, _ = ev.build_v2(ckpt.get("config", {}), device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    masked_cache = torch.load(args.sym_esm_cache, map_location="cpu", weights_only=False)

    dataset = dict(MegaScaleTestDatasets(data_root=args.data_root).iter_named())[args.benchmark]
    violations, fwd_all, rev_all = [], [], []
    for pidx in range(min(len(dataset), args.n)):
        protein = dataset[pidx]
        if protein is None:
            continue
        num_muts = len(protein.get("mut_ids", []))
        for mi in range(num_muts):
            p1 = copy.deepcopy(protein)
            p1["mut_ids"] = protein["mut_ids"][mi:mi+1]
            p1["append_tensors"] = protein["append_tensors"][mi:mi+1]
            if torch.is_tensor(protein["ddG"]) and protein["ddG"].dim() > 0:
                p1["ddG"] = protein["ddG"][mi:mi+1]
            elif not torch.is_tensor(protein["ddG"]) and hasattr(protein["ddG"], "__len__") and len(protein["ddG"]) > 1:
                p1["ddG"] = protein["ddG"][mi:mi+1]
            fwd = score(model, p1, masked_cache, device)
            rev = score(model, apply_mutation_to_S(p1), masked_cache, device)
            if fwd is None or rev is None or len(fwd) == 0 or len(rev) == 0:
                continue
            violations.append(abs(fwd[0] + rev[0]))
            fwd_all.append(fwd[0]); rev_all.append(rev[0])

    v = np.array(violations)
    print(f"\n=== independent-input anti-symmetry on {args.benchmark} (n={len(v)}) ===")
    print(f"|ΔΔĜ_fwd + ΔΔĜ_rev|:  mean={v.mean():.3e}  max={v.max():.3e}  median={np.median(v):.3e}")
    if v.max() < 1e-4:
        print("=> EXACT anti-symmetry under independent inputs (violation at float precision). ✓")
    elif v.mean() < 1e-2:
        print("=> Near-exact; small residual — check masked-cache coverage (some sites may hit zero-fallback).")
    else:
        print("=> NOT anti-symmetric — masked cache likely missing (keys not matching) -> fell back to zeros/unmasked.")
        print("   Verify masked_seq_key matches between precompute and adapter.")


if __name__ == "__main__":
    main()
