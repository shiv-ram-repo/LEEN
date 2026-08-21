#!/usr/bin/env python3
"""Precompute SYMMETRIZED ESM-2 embeddings for exact anti-symmetry WITHOUT information loss.

Novelty: instead of MASKING the mutation site (which discards information and costs accuracy), we
store the AVERAGE of the wild-type and single-mutant sequence embeddings:
    E_sym = ½( ESM(seq_wt) + ESM(seq_mut) )
Averaging is symmetric in (wt, mut), so the forward context (built from seq_wt) and the reverse
context (built from seq_mut) retrieve the SAME embedding -> exact anti-symmetry by construction,
while FULL evolutionary information is preserved (both WT and mutant context are present).

Key: masked_seq_key(S_np, pos) — same key the adapter uses; WT and mutant collapse to one key
because the key masks the site with a sentinel (identity-independent).

  PYTHONPATH=/backup/new_work python precompute_esm_symmetric.py \
    --data_root /backup/stability/data_root --fp16 --output sym_esm_cache_full.pt
"""
from __future__ import annotations
import argparse, logging
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sym_esm")

from leen.data.adapter_v2_masked import masked_seq_key, ALPHABET_21
ALPHA21_LIST = list(ALPHABET_21)


def s_to_str(S_np):
    return "".join(ALPHABET_21[i] if i < 21 else "X" for i in S_np)


def collect(data_root, benchmarks_only=False):
    """key -> (seq_wt_str, pos, mut_aa_char). One entry per (protein,mutation)."""
    from protstab_data import MegaScaleTestDatasets, MegaScaleDataset
    out = {}
    def add(dataset):
        for i in range(len(dataset)):
            p = dataset[i]
            if p is None:
                continue
            S = np.asarray(p["S"]).reshape(-1)
            seq = s_to_str(S)
            at = np.asarray(p["append_tensors"])
            for mi, m in enumerate(p.get("mut_ids", [])):
                pos = int(m)
                if not (0 <= pos < len(S)):
                    continue
                mt_leen = int(at[mi, 21:].argmax())
                mt_char = ALPHA21_LIST[mt_leen] if mt_leen < 21 else "X"
                out[masked_seq_key(S, pos)] = (seq, pos, mt_char)
    coll = MegaScaleTestDatasets(data_root=data_root)
    for _, ds in coll.iter_named():
        add(ds)
    if not benchmarks_only:
        try:
            add(MegaScaleDataset(data_root=data_root))
        except Exception as e:
            log.warning(f"train set not added ({e}); benchmarks only")
    return out


@torch.no_grad()
def embed(seq, model, alphabet, bc, device, n_layers):
    _, _, toks = bc([("x", seq[:1022])])
    toks = toks.to(device)
    rep = model(toks, repr_layers=[n_layers])["representations"][n_layers]
    return rep[0, 1:1 + min(len(seq), 1022), :].cpu()


@torch.no_grad()
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root", required=True)
    pa.add_argument("--output", default="sym_esm_cache_full.pt")
    pa.add_argument("--benchmarks_only", action="store_true")
    pa.add_argument("--fp16", action="store_true")
    args = pa.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()
    n_layers = model.num_layers

    items = collect(args.data_root, args.benchmarks_only)
    log.info(f"{len(items)} (protein,mutation) symmetrized embeddings to compute")

    wt_cache = {}
    cache = {}
    for k, (seq, pos, mt_char) in tqdm(items.items(), desc="sym ESM"):
        if pos >= len(seq[:1022]):
            continue
        if seq not in wt_cache:
            wt_cache[seq] = embed(seq, model, alphabet, bc, device, n_layers)
        e_wt = wt_cache[seq]
        mut_seq = seq[:pos] + mt_char + seq[pos + 1:]
        e_mut = embed(mut_seq, model, alphabet, bc, device, n_layers)
        L = min(e_wt.shape[0], e_mut.shape[0])
        e_sym = 0.5 * (e_wt[:L] + e_mut[:L])
        cache[k] = e_sym.half().clone() if args.fp16 else e_sym.clone()
    torch.save(cache, args.output)
    log.info(f"wrote {len(cache)} symmetrized embeddings -> {args.output}")


if __name__ == "__main__":
    main()
