#!/usr/bin/env python3
"""
G2 ablation: ESM-concat (no gating, no cross-attention).

ESM embeddings are projected to 128-dim and concatenated with the 13-dim
geometry features → 141-dim node features fed to a plain EGNN.
No gating, no cross-attention. Tests the baseline "just add ESM as features"
approach to prove that the gating mechanism matters.

Usage:
    PYTHONPATH=/backup/new_work:$PYTHONPATH python train_g2_concat.py \
        --data_root /backup/new_work/data_root \
        --esm_cache esm_cache_full.pt \
        --output_dir runs/leen_g2_concat \
        --lr 1e-4 --seed 42
"""
import argparse, hashlib, json, logging, os, time, random
import numpy as np, torch, torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from protstab_data import MegaScaleDataset, MegaScaleTestDatasets
from leen.models.energy_model import LEEN
from leen.losses import LEENLoss
from leen.data.adapter_v2 import protein_to_graphs_v2, collate_v2, ALPHA21_TO_LEEN
from leen.utils import compute_metrics, seed_everything, setup_logging

log = logging.getLogger(__name__)

ALPHABET_21 = "ACDEFGHIKLMNPQRSTVWYX"


class LEENConcat(nn.Module):
    """LEEN with ESM concatenated as node features.

    ESM 1280 → project to 128 → concatenate with 13-dim geo → 141-dim input.
    Plain EGNN, no gating, no cross-attention.
    Energy-difference readout (exact antisymmetry preserved).
    """
    def __init__(self, esm_dim=1280, esm_proj_dim=128, **leen_kwargs):
        super().__init__()
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, esm_proj_dim),
            nn.SiLU(),
            nn.LayerNorm(esm_proj_dim),
        )
        self.esm_proj_dim = esm_proj_dim
        self.leen = LEEN(input_node_dim=13 + esm_proj_dim, **leen_kwargs)

    def forward(self, graph):
        if graph.esm_features is not None:
            esm_proj = self.esm_proj(graph.esm_features)
        else:
            esm_proj = torch.zeros(
                graph.node_features.size(0), self.esm_proj_dim,
                device=graph.node_features.device,
            )

        from leen.models.energy_model import LocalGraph
        concat_graph = LocalGraph(
            node_features=torch.cat([graph.node_features, esm_proj], dim=-1),
            coords=graph.coords,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            aa_indices=graph.aa_indices,
            mut_pos=graph.mut_pos,
            wt_aa=graph.wt_aa,
            mut_aa=graph.mut_aa,
            ddg=graph.ddg,
            batch=graph.batch,
        )
        return self.leen(concat_graph)

    def predict_with_perturbation(self, graph, sigma=0.2):
        if graph.esm_features is not None:
            esm_proj = self.esm_proj(graph.esm_features)
        else:
            esm_proj = torch.zeros(
                graph.node_features.size(0), self.esm_proj_dim,
                device=graph.node_features.device,
            )

        from leen.models.energy_model import LocalGraph
        concat_graph = LocalGraph(
            node_features=torch.cat([graph.node_features, esm_proj], dim=-1),
            coords=graph.coords,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            aa_indices=graph.aa_indices,
            mut_pos=graph.mut_pos,
            wt_aa=graph.wt_aa,
            mut_aa=graph.mut_aa,
            ddg=graph.ddg,
            batch=graph.batch,
        )
        return self.leen.predict_with_perturbation(concat_graph, sigma=sigma)


def drop_esm(graphs, p):
    if random.random() < p:
        for g in graphs:
            g.esm_features = None
    return graphs


def train_epoch(model, ds, esm_cache, crit, opt, dev, cfg):
    model.train()
    tot, n = 0.0, 0
    for idx in tqdm(np.random.permutation(len(ds)), desc="Train", leave=False):
        p = ds[int(idx)]
        if p is None: continue
        try:
            gs = protein_to_graphs_v2(p, esm_cache, 10.0, 8.0, 64)
            if not gs: continue
            gs = drop_esm(gs, cfg.esm_dropout)
            b = collate_v2(gs).to(dev)
            if b.ddg is None: continue
            opt.zero_grad()
            if cfg.ood:
                o = model.predict_with_perturbation(b, sigma=0.2)
                ld = crit(o["ddg_pred"], b.ddg, o["ddg_perturbed"])
            else:
                o = model(b)
                ld = crit(o["ddg_pred"], b.ddg)
            if torch.isnan(ld["total"]) or torch.isinf(ld["total"]):
                log.warning(f"NaN at {idx}, skip"); opt.zero_grad(); continue
            ld["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += ld["total"].item(); n += 1
        except Exception as e:
            log.warning(f"Skip {idx}: {e}"); torch.cuda.empty_cache()
    return tot / max(n, 1)


@torch.no_grad()
def eval_ds(model, ds, esm_cache, dev, name=""):
    model.eval(); preds, tgts = [], []
    for idx in tqdm(range(len(ds)), desc=f"Eval {name}", leave=False):
        try:
            p = ds[idx]
            if p is None: continue
            gs = protein_to_graphs_v2(p, esm_cache, 10.0, 8.0, 64)
            if not gs: continue
            b = collate_v2(gs).to(dev)
            if b.ddg is None: continue
            preds.append(model(b)["ddg_pred"].cpu().numpy())
            tgts.append(b.ddg.cpu().numpy())
        except: torch.cuda.empty_cache(); continue
    if not preds: return {"spearman": 0.0, "n": 0}
    m = compute_metrics(np.concatenate(preds), np.concatenate(tgts))
    m["n"] = sum(len(x) for x in preds); return m


class Cfg:
    def __init__(self, **kw):
        self.esm_dropout = kw.get("esm_dropout", 0.25)
        self.ood = kw.get("ood", True)
        self.lr = kw.get("lr", 1e-4)
        self.seed = kw.get("seed", 42)
        self.max_epochs = kw.get("max_epochs", 200)
        self.patience = kw.get("patience", 30)
        self.loss_type = kw.get("loss_type", "bmc")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root", required=True)
    pa.add_argument("--esm_cache", required=True)
    pa.add_argument("--output_dir", default="runs/leen_g2_concat")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--max_epochs", type=int, default=200)
    pa.add_argument("--esm_dropout", type=float, default=0.25)
    args = pa.parse_args()

    cfg = Cfg(seed=args.seed, lr=args.lr, max_epochs=args.max_epochs,
              esm_dropout=args.esm_dropout)

    setup_logging(); seed_everything(cfg.seed)
    save_dir = os.path.join(args.output_dir, f"seed_{cfg.seed}")
    os.makedirs(save_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"G2 ablation: ESM-concat (no gating, no cross-attention)")
    log.info(f"Loading ESM cache: {args.esm_cache}")
    esm_cache = torch.load(args.esm_cache, map_location="cpu", weights_only=False)
    log.info(f"  {len(esm_cache)} sequences")

    train_ds = MegaScaleDataset(data_root=args.data_root, split="train")
    val_ds = MegaScaleDataset(data_root=args.data_root, split="val")

    model = LEENConcat(
        esm_dim=1280, esm_proj_dim=128,
        aa_embed_dim=64, node_dim=128, edge_dim=16,
        hidden_dim=256, num_layers=6, dropout=0.1,
        energy_hidden_dim=256,
    ).to(dev)
    log.info(f"LEENConcat: {sum(p.numel() for p in model.parameters()):,} params")

    crit = LEENLoss(primary="bmc", bmc_sigma_init=1.0, ood_weight=0.5).to(dev)
    opt = AdamW(list(model.parameters()) + list(crit.parameters()),
                lr=cfg.lr, weight_decay=1e-4)

    best, pat = -1.0, 0
    for ep in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_ds, esm_cache, crit, opt, dev, cfg)
        vm = eval_ds(model, val_ds, esm_cache, dev, "val")
        log.info(f"Epoch {ep:3d} | loss={loss:.4f} | val_sp={vm['spearman']:.4f} | {time.time()-t0:.1f}s")
        if vm["spearman"] > best:
            best = vm["spearman"]; pat = 0
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                         "val_spearman": best},
                        os.path.join(save_dir, "best.pt"))
            log.info(f"  ✓ New best: {best:.4f}")
        else:
            pat += 1
            if pat >= cfg.patience: log.info(f"Early stop at {ep}"); break

    ckpt = torch.load(os.path.join(save_dir, "best.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    coll = MegaScaleTestDatasets(data_root=args.data_root)
    res = {}
    log.info(f"\n{'Benchmark':18s} {'n':>6s}  {'Spearman':>8s}")
    log.info("-" * 40)
    for name, ds in coll.iter_named():
        m = eval_ds(model, ds, esm_cache, dev, name)
        res[name] = m; log.info(f"{name:18s} {m['n']:6d}  {m['spearman']:8.4f}")
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    log.info("Done!")

if __name__ == "__main__":
    main()
