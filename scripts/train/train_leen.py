#!/usr/bin/env python3
"""
LEEN v2 training with ESM dropout.

Key change from train_leen_v2.py: during training, ESM features are
randomly zeroed out for --esm_dropout fraction of proteins (default 25%).
This forces the structural branch to learn independently and prevents
catastrophic failure when ESM cache misses occur at eval time.

Usage:
    PYTHONPATH=/backup/new_work:$PYTHONPATH python train_leen_v2_drop.py \
        --data_root /backup/new_work/data_root \
        --esm_cache esm_cache_full.pt \
        --output_dir runs/leen_v2_drop \
        --esm_dropout 0.25 \
        --seed 42
"""
import argparse, json, logging, os, time, random
from dataclasses import dataclass
import numpy as np, torch
from torch.optim import AdamW
from tqdm import tqdm

from protstab_data import MegaScaleDataset, MegaScaleTestDatasets
from leen.models.leen_v2 import LEENv2
from leen.losses import LEENLoss
from leen.data.adapter_v2 import protein_to_graphs_v2, collate_v2
from leen.utils import compute_metrics, seed_everything, setup_logging

log = logging.getLogger(__name__)


@dataclass
class Cfg:
    data_root: str = ""
    esm_cache_path: str = ""
    output_dir: str = "runs/leen_v2_drop"
    geo_dim: int = 13; esm_dim: int = 1280; esm_proj_dim: int = 128
    aa_embed_dim: int = 64; node_dim: int = 128; hidden_dim: int = 256
    num_layers: int = 6; dropout: float = 0.1; energy_hidden_dim: int = 256
    use_cross_attention: bool = True; cross_attn_heads: int = 4; encoder_type: str = "egnn"
    env_radius: float = 10.0; edge_cutoff: float = 8.0; max_neighbors: int = 64
    seed: int = 42; max_epochs: int = 200; patience: int = 20
    lr: float = 1e-4; wd: float = 1e-4; grad_clip: float = 1.0
    loss_type: str = "bmc"; bmc_sigma: float = 1.0
    ood: bool = True; ood_sigma: float = 0.20; ood_weight: float = 0.5
    esm_dropout: float = 0.25
    head_type: str = "energy"


def drop_esm_from_graphs(graphs, drop_prob):
    """Randomly zero out ESM features for a protein's graphs.

    Applied at the PROTEIN level, not per-mutation, so either all
    mutations of a protein have ESM or none do. This is more realistic
    since cache misses affect whole proteins.
    """
    if random.random() < drop_prob:
        for g in graphs:
            g.esm_features = None
    return graphs


def train_epoch(model, ds, esm_cache, crit, opt, dev, cfg):
    model.train()
    tot, n = 0.0, 0
    n_with_esm, n_without_esm = 0, 0

    for idx in tqdm(np.random.permutation(len(ds)), desc="Train", leave=False):
        p = ds[int(idx)]
        if p is None:
            continue
        try:
            gs = protein_to_graphs_v2(
                p, esm_cache, cfg.env_radius, cfg.edge_cutoff, cfg.max_neighbors,
            )
            if not gs:
                continue

            gs = drop_esm_from_graphs(gs, cfg.esm_dropout)

            if gs[0].esm_features is not None:
                n_with_esm += 1
            else:
                n_without_esm += 1

            b = collate_v2(gs).to(dev)
            if b.ddg is None:
                continue

            opt.zero_grad()
            if cfg.ood:
                o = model.predict_with_perturbation(b, sigma=cfg.ood_sigma)
                ld = crit(o["ddg_pred"], b.ddg, o["ddg_perturbed"])
            else:
                o = model(b)
                ld = crit(o["ddg_pred"], b.ddg)

            if torch.isnan(ld["total"]) or torch.isinf(ld["total"]):
                log.warning(f"NaN loss at protein {idx}, skipping")
                opt.zero_grad()
                torch.cuda.empty_cache()
                continue
            ld["total"].backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += ld["total"].item()
            n += 1

        except Exception as e:
            log.warning(f"Skip {idx}: {e}")
            torch.cuda.empty_cache()

    pct = 100 * n_with_esm / max(n_with_esm + n_without_esm, 1)
    return tot / max(n, 1), pct


@torch.no_grad()
def eval_ds(model, ds, esm_cache, dev, cfg, name=""):
    model.eval()
    preds, tgts = [], []
    for idx in tqdm(range(len(ds)), desc=f"Eval {name}", leave=False):
        try:
            p = ds[idx]
            if p is None:
                continue
            gs = protein_to_graphs_v2(
                p, esm_cache, cfg.env_radius, cfg.edge_cutoff, cfg.max_neighbors,
            )
            if not gs:
                continue
            b = collate_v2(gs).to(dev)
            if b.ddg is None:
                continue
            preds.append(model(b)["ddg_pred"].cpu().numpy())
            tgts.append(b.ddg.cpu().numpy())
        except Exception:
            torch.cuda.empty_cache()
            continue
    if not preds:
        return {"spearman": 0.0, "n": 0}
    m = compute_metrics(np.concatenate(preds), np.concatenate(tgts))
    m["n"] = sum(len(x) for x in preds)
    return m


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root", required=True)
    pa.add_argument("--esm_cache", required=True)
    pa.add_argument("--output_dir", default="runs/leen_v2_drop")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--loss", default="bmc")
    pa.add_argument("--no_ood", action="store_true")
    pa.add_argument("--no_cross_attn", action="store_true")
    pa.add_argument("--encoder", default="egnn", choices=["egnn", "gvp"])
    pa.add_argument("--max_epochs", type=int, default=200)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--head_type", default="energy", choices=["energy", "scalar"])
    pa.add_argument("--esm_dropout", type=float, default=0.25,
                    help="Fraction of training proteins with ESM dropped (default 0.25)")
    args = pa.parse_args()

    cfg = Cfg(
        data_root=args.data_root, esm_cache_path=args.esm_cache,
        output_dir=args.output_dir, seed=args.seed, loss_type=args.loss,
        ood=not args.no_ood, use_cross_attention=not args.no_cross_attn,
        max_epochs=args.max_epochs, lr=args.lr, esm_dropout=args.esm_dropout,
        encoder_type=args.encoder, head_type=args.head_type,
    )

    setup_logging()
    seed_everything(cfg.seed)
    save_dir = os.path.join(cfg.output_dir, f"seed_{cfg.seed}")
    os.makedirs(save_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Device: {dev} | Seed: {cfg.seed} | ESM dropout: {cfg.esm_dropout}")

    log.info(f"Loading ESM cache: {cfg.esm_cache_path}")
    esm_cache = torch.load(cfg.esm_cache_path, map_location="cpu", weights_only=False)
    log.info(f"  {len(esm_cache)} sequences cached")

    log.info("Loading Megascale...")
    train_ds = MegaScaleDataset(data_root=cfg.data_root, split="train")
    val_ds = MegaScaleDataset(data_root=cfg.data_root, split="val")
    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = LEENv2(
        geo_dim=cfg.geo_dim, esm_dim=cfg.esm_dim, esm_proj_dim=cfg.esm_proj_dim,
        aa_embed_dim=cfg.aa_embed_dim, node_dim=cfg.node_dim, edge_dim=16,
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, dropout=cfg.dropout,
        energy_hidden_dim=cfg.energy_hidden_dim,
        use_cross_attention=cfg.use_cross_attention,
        cross_attn_heads=cfg.cross_attn_heads,
        encoder_type=cfg.encoder_type,
        head_type=cfg.head_type,
    ).to(dev)
    log.info(f"LEENv2: {sum(p.numel() for p in model.parameters()):,} params")

    crit = LEENLoss(
        primary=cfg.loss_type, bmc_sigma_init=cfg.bmc_sigma,
        ood_weight=cfg.ood_weight if cfg.ood else 0.0,
    ).to(dev)
    opt = AdamW(
        list(model.parameters()) + list(crit.parameters()),
        lr=cfg.lr, weight_decay=cfg.wd,
    )

    best, pat = -1.0, 0
    for ep in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        loss, esm_pct = train_epoch(model, train_ds, esm_cache, crit, opt, dev, cfg)
        vm = eval_ds(model, val_ds, esm_cache, dev, cfg, "val")
        log.info(
            f"Epoch {ep:3d} | loss={loss:.4f} | val_sp={vm['spearman']:.4f} | "
            f"esm_used={esm_pct:.0f}% | {time.time()-t0:.1f}s"
        )
        if vm["spearman"] > best:
            best = vm["spearman"]
            pat = 0
            torch.save({
                "epoch": ep, "model_state": model.state_dict(),
                "val_spearman": best, "config": cfg.__dict__,
            }, os.path.join(save_dir, "best.pt"))
            log.info(f"  ✓ New best: {best:.4f}")
        else:
            pat += 1
            if pat >= cfg.patience:
                log.info(f"Early stop at {ep}")
                break

    ckpt = torch.load(os.path.join(save_dir, "best.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    coll = MegaScaleTestDatasets(data_root=cfg.data_root)
    res = {}
    log.info(f"\n{'Benchmark':18s} {'n':>6s}  {'Spearman':>8s}")
    log.info("-" * 40)
    for name, ds in coll.iter_named():
        m = eval_ds(model, ds, esm_cache, dev, cfg, name)
        res[name] = m
        log.info(f"{name:18s} {m['n']:6d}  {m['spearman']:8.4f}")
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    log.info("Done!")


if __name__ == "__main__":
    main()
