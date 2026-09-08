#!/usr/bin/env python3
"""LEEN per-sample evaluator (all model families) -> single source of truth for the revision.

Dispatches on the checkpoint's saved config to build the RIGHT model + graph pipeline, exactly as
each training script does (read from train_leen.py / train_leen_v2_drop.py / train_g2_concat.py):

  family 'v1'     : LEEN (energy_model) + base adapter (protein_dict_to_local_graphs/collate_local_graphs)
  family 'v2'     : LEENv2 + adapter_v2 (protein_to_graphs_v2/collate_v2), needs ESM cache
  family 'concat' : LEENConcat wrapper (from train_g2_concat) + adapter_v2, needs ESM cache

Writes per-sample rows with a protein cluster id for protein-clustered bootstrap:
  <out_dir>/<benchmark>.csv  columns: protein_idx, mut_index, pred, true

  PYTHONPATH=/backup/new_work python eval_leen_persample.py \
    --ckpt runs/leen_v2_drop/seed_42/best.pt --family v2 \
    --data_root /backup/stability/data_root \
    --esm_cache /backup/new_work/leen/esm_cache_full.pt \
    --out_dir runs/leen_preds/leen_v2_drop_seed42
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
import pandas as pd
import torch
from tqdm import tqdm

from protstab_data import MegaScaleTestDatasets


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_v1(cfg, device):
    from leen.models import LEEN
    from leen.data.adapter import protein_dict_to_local_graphs, collate_local_graphs
    m = LEEN(input_node_dim=13, aa_embed_dim=_cfg_get(cfg, "aa_embed_dim", 64),
             node_dim=_cfg_get(cfg, "node_dim", 128), edge_dim=16,
             hidden_dim=_cfg_get(cfg, "hidden_dim", 256), num_layers=_cfg_get(cfg, "num_layers", 6),
             dropout=0.0, energy_hidden_dim=_cfg_get(cfg, "energy_hidden_dim", 256)).to(device)
    def build_graphs(p, esm_cache):
        return protein_dict_to_local_graphs(p, env_radius=_cfg_get(cfg, "env_radius", 10.0),
                                            edge_cutoff=_cfg_get(cfg, "edge_cutoff", 8.0),
                                            max_neighbors=_cfg_get(cfg, "max_neighbors", 64))
    return m, build_graphs, collate_local_graphs


def build_v2(cfg, device):
    from leen.models.leen_v2 import LEENv2
    from leen.data.adapter_v2 import protein_to_graphs_v2, collate_v2
    m = LEENv2(geo_dim=_cfg_get(cfg, "geo_dim", 13), esm_dim=_cfg_get(cfg, "esm_dim", 1280),
               esm_proj_dim=_cfg_get(cfg, "esm_proj_dim", 128), aa_embed_dim=_cfg_get(cfg, "aa_embed_dim", 64),
               node_dim=_cfg_get(cfg, "node_dim", 128), edge_dim=16,
               hidden_dim=_cfg_get(cfg, "hidden_dim", 256), num_layers=_cfg_get(cfg, "num_layers", 6),
               dropout=0.0, energy_hidden_dim=_cfg_get(cfg, "energy_hidden_dim", 256),
               use_cross_attention=_cfg_get(cfg, "use_cross_attention", True),
               cross_attn_heads=_cfg_get(cfg, "cross_attn_heads", 4),
               encoder_type=_cfg_get(cfg, "encoder_type", "egnn"),
               symmetrize_site=_cfg_get(cfg, "symmetrize_site", False),
               head_type=_cfg_get(cfg, "head_type", "energy")).to(device)
    def build_graphs(p, esm_cache):
        return protein_to_graphs_v2(p, esm_cache, _cfg_get(cfg, "env_radius", 10.0),
                                    _cfg_get(cfg, "edge_cutoff", 8.0), _cfg_get(cfg, "max_neighbors", 64))
    return m, build_graphs, collate_v2


def build_v2_symmetric(cfg, device, sym_cache):
    """Same LEENv2, graphs built with the SYMMETRIC adapter + symmetrized ESM cache
    (½(E_wt+E_mut)) — for models trained with feature symmetrization (exact anti-symmetry, full info)."""
    from leen.data.adapter_v2_symmetric import protein_to_graphs_v2_symmetric
    from leen.data.adapter_v2 import collate_v2
    m, _, _ = build_v2(cfg, device)
    def build_graphs(p, esm_cache):
        return protein_to_graphs_v2_symmetric(
            p, esm_cache=esm_cache, sym_esm_cache=sym_cache,
            env_radius=_cfg_get(cfg, "env_radius", 10.0),
            edge_cutoff=_cfg_get(cfg, "edge_cutoff", 8.0),
            max_neighbors=_cfg_get(cfg, "max_neighbors", 64))
    return m, build_graphs, collate_v2


def build_v2_masked(cfg, device, masked_cache):
    """Same LEENv2, but graphs are built with the MASKED adapter + masked ESM cache
    (needed to evaluate a model trained with mutation-site masking — train/eval must match)."""
    from leen.data.adapter_v2_masked import protein_to_graphs_v2_masked
    from leen.data.adapter_v2 import collate_v2
    m, _, _ = build_v2(cfg, device)
    def build_graphs(p, esm_cache):
        return protein_to_graphs_v2_masked(
            p, esm_cache=esm_cache, masked_esm_cache=masked_cache,
            env_radius=_cfg_get(cfg, "env_radius", 10.0),
            edge_cutoff=_cfg_get(cfg, "edge_cutoff", 8.0),
            max_neighbors=_cfg_get(cfg, "max_neighbors", 64))
    return m, build_graphs, collate_v2


def build_concat(cfg, device, train_script):
    """Import LEENConcat from its training script (module-level class)."""
    spec = importlib.util.spec_from_file_location("train_g2_concat", train_script)
    mod = importlib.util.module_from_spec(spec); sys.modules["train_g2_concat"] = mod
    spec.loader.exec_module(mod)
    from leen.data.adapter_v2 import protein_to_graphs_v2, collate_v2
    m = mod.LEENConcat(esm_dim=_cfg_get(cfg, "esm_dim", 1280), esm_proj_dim=_cfg_get(cfg, "esm_proj_dim", 128),
                       aa_embed_dim=_cfg_get(cfg, "aa_embed_dim", 64), node_dim=_cfg_get(cfg, "node_dim", 128),
                       hidden_dim=_cfg_get(cfg, "hidden_dim", 256), num_layers=_cfg_get(cfg, "num_layers", 6),
                       dropout=0.0, energy_hidden_dim=_cfg_get(cfg, "energy_hidden_dim", 256)).to(device)
    def build_graphs(p, esm_cache):
        return protein_to_graphs_v2(p, esm_cache, 10.0, 8.0, 64)
    return m, build_graphs, collate_v2


def detect_family(cfg):
    if _cfg_get(cfg, "use_cross_attention", None) is not None or _cfg_get(cfg, "geo_dim", None) is not None:
        return "v2"
    return "v1"


@torch.no_grad()
def eval_persample(model, dataset, build_graphs, collate, esm_cache, device):
    rows = []
    for pidx in tqdm(range(len(dataset)), leave=False):
        try:
            p = dataset[pidx]
            if p is None:
                continue
            gs = build_graphs(p, esm_cache)
            if not gs:
                continue
            b = collate(gs).to(device)
            if b.ddg is None:
                continue
            pred = model(b)["ddg_pred"].cpu().numpy().reshape(-1)
            true = b.ddg.cpu().numpy().reshape(-1)
            for j in range(min(len(pred), len(true))):
                rows.append((pidx, j, float(pred[j]), float(true[j])))
        except Exception:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
    return rows


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", required=True); pa.add_argument("--data_root", required=True)
    pa.add_argument("--out_dir", required=True)
    pa.add_argument("--family", choices=["auto", "v1", "v2", "concat"], default="auto")
    pa.add_argument("--esm_cache", default="", help="required for v2/concat families")
    pa.add_argument("--masked", action="store_true", help="use masked adapter (for masked-trained ckpts)")
    pa.add_argument("--symmetric", action="store_true", help="use symmetric adapter (for sym-trained ckpts)")
    pa.add_argument("--sym_esm_cache", default="", help="symmetrized ESM cache (required with --symmetric)")
    pa.add_argument("--masked_esm_cache", default="", help="masked ESM cache (required with --masked)")
    pa.add_argument("--concat_train_script", default="train_g2_concat.py")
    args = pa.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    fam = args.family if args.family != "auto" else detect_family(cfg)
    print(f"family={fam}  (epoch {ckpt.get('epoch')}, val_sp={ckpt.get('val_spearman')})")

    masked_cache = None
    if args.masked:
        assert args.masked_esm_cache, "--masked requires --masked_esm_cache"
        masked_cache = torch.load(args.masked_esm_cache, map_location="cpu", weights_only=False)
    sym_cache = None
    if args.symmetric:
        assert args.sym_esm_cache, "--symmetric requires --sym_esm_cache"
        sym_cache = torch.load(args.sym_esm_cache, map_location="cpu", weights_only=False)
    if fam == "v1":
        model, build_graphs, collate = build_v1(cfg, device)
    elif fam == "v2":
        if args.symmetric:
            model, build_graphs, collate = build_v2_symmetric(cfg, device, sym_cache)
        elif args.masked:
            model, build_graphs, collate = build_v2_masked(cfg, device, masked_cache)
        else:
            model, build_graphs, collate = build_v2(cfg, device)
    else:
        model, build_graphs, collate = build_concat(cfg, device, args.concat_train_script)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    print(f"loaded {fam}: {sum(p.numel() for p in model.parameters()):,} params")

    esm_cache = None
    if fam in ("v2", "concat"):
        assert args.esm_cache, "v2/concat need --esm_cache"
        print(f"loading ESM cache {args.esm_cache} ...")
        esm_cache = torch.load(args.esm_cache, map_location="cpu", weights_only=False)

    coll = MegaScaleTestDatasets(data_root=args.data_root)
    manifest = {}
    for name, dataset in coll.iter_named():
        rows = eval_persample(model, dataset, build_graphs, collate, esm_cache, device)
        df = pd.DataFrame(rows, columns=["protein_idx", "mut_index", "pred", "true"])
        df.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        manifest[name] = {"n": len(df), "n_proteins": int(df["protein_idx"].nunique()) if len(df) else 0}
        print(f"{name:18s} n={len(df):6d}  proteins={manifest[name]['n_proteins']:4d}")
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=2)
    print(f"\nper-sample predictions -> {args.out_dir}/")


if __name__ == "__main__":
    main()
