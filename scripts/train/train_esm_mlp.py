#!/usr/bin/env python3
"""Frozen-ESM-2 + MLP baseline (Reviewer 2 & 4): quantifies the gain of the structural-fusion
mechanism over the language-model representation ALONE. No structure — for each mutation the input
is the frozen ESM-2 embedding at the mutation site plus the WT/mutant identity; an MLP predicts ΔΔG.

Trains on the SAME Megascale training split (same MMseqs2 0.25 filter) and evaluates on the SAME
11 benchmarks, so the number is directly comparable to LEEN in the master table.

  PYTHONPATH=/backup/new_work python train_esm_mlp_baseline.py \
    --data_root /backup/stability/data_root \
    --esm_cache /backup/new_work/leen/esm_cache_full.pt \
    --out_dir /backup/new_work/runs/esm_mlp --seed 42
"""
from __future__ import annotations
import argparse, os, logging
import numpy as np, torch, torch.nn as nn
from tqdm import tqdm

logging.basicConfig(level=logging.INFO); log = logging.getLogger("esm_mlp")

from leen.data.adapter_v2 import _seq_key


def featurize(dataset, esm_cache, desc):
    """For each mutation: x = [esm_site (1280), wt_onehot(21), mut_onehot(21)], y = ddG.
    Real data layout (verified): S is (1,L); ONE protein dict holds M mutations with
    append_tensors [M,42] (21 wt + 21 mut one-hot), ddG [M,1], mut_ids a length-M list of positions.
    Cache key = _seq_key(S) (md5 hash) from the working adapter."""
    X, Y = [], []
    for i in tqdm(range(len(dataset)), desc=desc):
        try:
            p = dataset[i]
            if p is None: continue
            S = np.asarray(p["S"]).reshape(-1)
            key = _seq_key(S)
            if key not in esm_cache: continue
            emb = esm_cache[key].float()
            at = p["append_tensors"]
            muts = p["mut_ids"]
            ddg = p["ddG"]
            M = at.shape[0]
            for mi in range(M):
                pos = int(muts[mi])
                if pos < 0 or pos >= emb.shape[0]: continue
                y = ddg[mi]
                yv = float(y.reshape(-1)[0]) if hasattr(y, "reshape") else float(y)
                if yv != yv: continue
                site = emb[pos]
                wt = at[mi, :21].float(); mt = at[mi, 21:].float()
                X.append(torch.cat([site, wt, mt]).numpy())
                Y.append(yv)
        except Exception:
            continue
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


class MLP(nn.Module):
    def __init__(self, d_in, hidden=512, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root", required=True); pa.add_argument("--esm_cache", required=True)
    pa.add_argument("--out_dir", required=True); pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--epochs", type=int, default=100); pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--hidden", type=int, default=512); pa.add_argument("--bs", type=int, default=512)
    args = pa.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    from protstab_data import MegaScaleDataset, MegaScaleTestDatasets
    esm_cache = torch.load(args.esm_cache, map_location="cpu", weights_only=False)

    log.info("featurizing training set (frozen ESM site embeddings)...")
    Xtr, Ytr = featurize(MegaScaleDataset(data_root=args.data_root, split="train"), esm_cache, "train")
    Xva, Yva = featurize(MegaScaleDataset(data_root=args.data_root, split="val"), esm_cache, "val")
    log.info(f"train {Xtr.shape} val {Xva.shape}")

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    def norm(x): return (x - mu) / sd
    Xtr_t = torch.tensor(norm(Xtr)); Ytr_t = torch.tensor(Ytr)
    Xva_t = torch.tensor(norm(Xva)).to(device)

    model = MLP(Xtr.shape[1], args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss()

    best, best_state = -1, None
    n = len(Xtr_t)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n)
        for j in range(0, n, args.bs):
            idx = perm[j:j+args.bs]
            xb = Xtr_t[idx].to(device); yb = Ytr_t[idx].to(device)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xva_t).cpu().numpy()
        vs = spearman(vp, Yva)
        if vs > best:
            best = vs; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0: log.info(f"epoch {ep} val_sp={vs:.4f} (best {best:.4f})")

    model.load_state_dict(best_state)
    torch.save({"model_state": best_state, "mu": mu, "sd": sd, "val_sp": best}, os.path.join(args.out_dir, f"seed_{args.seed}.pt"))
    log.info(f"best val_sp={best:.4f}")

    coll = MegaScaleTestDatasets(data_root=args.data_root)
    preds_root = os.path.join(args.out_dir, f"ESMMLP_seed{args.seed}")
    os.makedirs(preds_root, exist_ok=True)
    print(f"\n{'benchmark':16s} {'n':>7s} {'Spearman':>10s}")
    for name, ds in coll.iter_named():
        Xb, Yb = featurize(ds, esm_cache, name)
        if len(Xb) < 5:
            print(f"{name:16s} {0:7d} {'--':>10s}"); continue
        with torch.no_grad():
            pb = model(torch.tensor(norm(Xb)).to(device)).cpu().numpy()
        sp = spearman(pb, Yb)
        print(f"{name:16s} {len(Xb):7d} {sp:10.4f}")
        import csv
        with open(os.path.join(preds_root, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["protein_idx","mut_index","pred","true"])
            for k in range(len(pb)): w.writerow([0, k, float(pb[k]), float(Yb[k])])
    log.info(f"per-sample predictions -> {preds_root}")


if __name__ == "__main__":
    main()
