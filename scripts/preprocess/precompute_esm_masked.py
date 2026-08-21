#!/usr/bin/env python3
"""Precompute SITE-MASKED ESM-2 embeddings for true anti-symmetry.

For every (sequence, mutation_position) that appears in training + benchmarks, run ESM-2 with that
position replaced by <mask>, and store the resulting per-residue embedding row at the site. Because
the site is masked, this embedding is IDENTICAL whether the true residue is wt or mut -> the model's
context becomes identity-independent -> exact anti-symmetry.

Key format:  "<seq_key>@<mut_pos>"  ->  Tensor[1280]  (the masked-site row only; that's all we need)

  PYTHONPATH=/backup/new_work python precompute_esm_masked.py \
    --data_root /backup/stability/data_root --output masked_esm_cache.pt
"""
from __future__ import annotations
import argparse, logging
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mask_esm")

from leen.data.adapter_v2_masked import masked_seq_key, ALPHABET_21


def s_to_str(S_np):
    """Reconstruct the ESM input string from the encoded S, matching the adapter's decoding."""
    return "".join(ALPHABET_21[i] if i < 21 else "X" for i in S_np)


def collect_seq_positions(data_root, benchmarks_only=False):
    """Every (sequence, mut_pos) pair across MegaScale train + all benchmarks."""
    from protstab_data import MegaScaleTestDatasets, MegaScaleDataset
    pairs = {}
    def add(dataset):
        for i in range(len(dataset)):
            p = dataset[i]
            if p is None:
                continue
            S = np.asarray(p["S"]).reshape(-1)
            seq_str = s_to_str(S)
            for m in p.get("mut_ids", []):
                pos = int(m)
                if 0 <= pos < len(S):
                    pairs[masked_seq_key(S, pos)] = (seq_str, pos)
    coll = MegaScaleTestDatasets(data_root=data_root)
    for _, ds in coll.iter_named():
        add(ds)
    if not benchmarks_only:
        try:
            add(MegaScaleDataset(data_root=data_root))
        except Exception as e:
            log.warning(f"train set not added ({e}); benchmarks only")
    return pairs


@torch.no_grad()
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root", required=True)
    pa.add_argument("--output", default="masked_esm_cache.pt")
    pa.add_argument("--benchmarks_only", action="store_true", help="skip training set (fast verification slice)")
    pa.add_argument("--fp16", action="store_true", help="store embeddings in float16 (halves disk)")
    args = pa.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()
    n_layers = model.num_layers

    pairs = collect_seq_positions(args.data_root, args.benchmarks_only)
    log.info(f"{len(pairs)} (masked seq,pos) embeddings to compute")

    cache = {}
    for k, (seq_full, pos) in tqdm(pairs.items(), desc="masked ESM"):
        seq = seq_full[:1022]
        if pos >= len(seq):
            continue
        _, _, tokens = bc([("x", seq)])
        tokens = tokens.to(device)
        tokens[0, 1 + pos] = alphabet.mask_idx
        rep = model(tokens, repr_layers=[n_layers])["representations"][n_layers]
        emb = rep[0, 1:1 + min(len(seq), 1022), :].cpu()
        cache[k] = emb.half().clone() if args.fp16 else emb.clone()
    torch.save(cache, args.output)
    log.info(f"wrote {len(cache)} masked-site embeddings -> {args.output}")


if __name__ == "__main__":
    main()
