#!/usr/bin/env python3
"""
Precompute ESM2 embeddings with FULL coverage.

The previous version parsed CSVs/JSONs and missed most benchmark proteins.
This version iterates through the actual dataset classes — the same ones
used during training and evaluation — so the sequence hash is guaranteed
to match what the adapter computes.

Usage:
    PYTHONPATH=/backup/new_work:$PYTHONPATH python precompute_esm_v2.py \
        --data_root /backup/new_work/data_root \
        --output esm_cache_full.pt
"""

import argparse
import hashlib
import logging

import torch
from tqdm import tqdm

from protstab_data import MegaScaleDataset, MegaScaleTestDatasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

ALPHABET_21 = "ACDEFGHIKLMNPQRSTVWYX"


def seq_key(S_tensor):
    """Exact same hash the adapter uses."""
    if S_tensor.dim() == 2:
        S_tensor = S_tensor.squeeze(0)
    S_np = S_tensor.cpu().numpy()
    seq = "".join(ALPHABET_21[i] if i < 21 else "X" for i in S_np)
    clean = seq.replace("-", "").replace("X", "")
    return hashlib.md5(clean.encode()).hexdigest(), seq


def collect_sequences(data_root):
    """Collect all unique sequences from every dataset class."""
    seqs = {}

    for split in ["train", "val"]:
        log.info(f"Scanning Megascale {split}...")
        ds = MegaScaleDataset(data_root=data_root, split=split)
        for idx in tqdm(range(len(ds)), desc=f"mega_{split}", leave=False):
            try:
                protein = ds[idx]
                if protein is None:
                    continue
                key, seq = seq_key(protein["S"])
                if key not in seqs:
                    seqs[key] = seq
            except Exception:
                continue

    log.info("Scanning benchmarks...")
    collection = MegaScaleTestDatasets(data_root=data_root)
    for name, dataset in collection.iter_named():
        log.info(f"  {name}: {len(dataset)} proteins")
        for idx in tqdm(range(len(dataset)), desc=name, leave=False):
            try:
                protein = dataset[idx]
                if protein is None:
                    continue
                key, seq = seq_key(protein["S"])
                if key not in seqs:
                    seqs[key] = seq
            except Exception:
                continue

    return seqs


def extract_embeddings(seqs, device, batch_size=4):
    """Run ESM2 on all unique sequences."""
    import esm

    log.info("Loading ESM2 (esm2_t33_650M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers

    cache = {}
    items = list(seqs.items())
    log.info(f"Extracting embeddings for {len(items)} sequences...")

    for i in tqdm(range(0, len(items), batch_size), desc="ESM2"):
        batch_items = items[i:i + batch_size]

        data = [(key, s[:1022]) for key, s in batch_items]

        try:
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            with torch.no_grad():
                results = model(tokens, repr_layers=[n_layers])
                reprs = results["representations"][n_layers]

            for j, (key, s) in enumerate(batch_items):
                full_len = len(s)
                trunc_len = min(full_len, 1022)
                emb = reprs[j, 1:1 + trunc_len, :].cpu()

                if emb.shape[0] < full_len:
                    emb = torch.cat([emb, torch.zeros(full_len - emb.shape[0], 1280)])

                cache[key] = emb

        except Exception as e:
            log.warning(f"Batch {i} failed: {e}")
            for key, s in batch_items:
                try:
                    data_single = [(key, s[:1022])]
                    _, _, tokens = batch_converter(data_single)
                    tokens = tokens.to(device)
                    with torch.no_grad():
                        results = model(tokens, repr_layers=[n_layers])
                        emb = results["representations"][n_layers][0, 1:1 + min(len(s), 1022), :].cpu()
                    if emb.shape[0] < len(s):
                        emb = torch.cat([emb, torch.zeros(len(s) - emb.shape[0], 1280)])
                    cache[key] = emb
                except Exception:
                    pass
            torch.cuda.empty_cache()

    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", default="esm_cache_full.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)

    seqs = collect_sequences(args.data_root)
    log.info(f"Total unique sequences: {len(seqs)}")

    cache = extract_embeddings(seqs, device, args.batch_size)
    log.info(f"Successfully embedded: {len(cache)} / {len(seqs)}")

    torch.save(cache, args.output)
    log.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
