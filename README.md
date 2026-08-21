# LEEN — Local Equivariant Energy Network

ΔΔG stability prediction via an ESM-gated equivariant energy network with an
energy-difference readout.

## Install

```
pip install -e .
pip install -r requirements.txt
```

## Data layout

Set `--data_root` to a directory containing:

```
data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv
data/dataset/megascale/mmseq_mut_search_0.25.m8
data/dataset/megascale/mega_splits.pkl
data/dataset/megascale/AlphaFold_model_PDBs/<wt_name>.pdb
data/dataset/<benchmark>/...            (S669, S461, S783, S2648, S8754, S571, S4346, Ssym, FireProt-HF)
```

## Pipeline

### 1. Precompute ESM-2 embeddings

```
python scripts/preprocess/precompute_esm.py --data_root DATA --output esm_cache_full.pt --fp16
```

### 2. Train

Main model (E2 configuration = ESM-gated EGNN + cross-attention + OOD loss):

```
python scripts/train/train_leen.py --data_root DATA --esm_cache esm_cache_full.pt \
  --output_dir runs/E2 --seed 42
```

Configuration flags:

- `--encoder {egnn,gvp,painn}`   geometric encoder (egnn = E2/E1, gvp = V1)
- `--no_cross_attn`              gating only, no cross-attention (E1)
- `--no_ood`                     disable the OOD consistency loss
- `--esm_dropout FLOAT`          fraction of training proteins with ESM dropped (default 0.25; set 0 to disable)
- `--head_type {energy,scalar}`  energy-difference readout (default) or direct scalar regression

Concatenation baseline (G2):

```
python scripts/train/train_concat.py --data_root DATA --esm_cache esm_cache_full.pt --output_dir runs/G2 --seed 42
```

Frozen-ESM + MLP baseline (no structure):

```
python scripts/train/train_esm_mlp.py --data_root DATA --esm_cache esm_cache_full.pt --out_dir runs/esm_mlp --seed 42
```

Exact anti-symmetry variants:

```
python scripts/preprocess/precompute_esm_symmetric.py --data_root DATA --output sym_esm_cache_full.pt --fp16
python scripts/train/train_symmetric.py --data_root DATA --esm_cache esm_cache_full.pt \
  --sym_esm_cache sym_esm_cache_full.pt --output_dir runs/symmetric --seed 42
```

### 3. Evaluate

```
python scripts/eval/eval_persample.py --ckpt runs/E2/seed_42/best.pt --family v2 \
  --esm_cache esm_cache_full.pt --data_root DATA --out_dir runs/preds/E2_seed42
```

For symmetric checkpoints add `--symmetric --sym_esm_cache sym_esm_cache_full.pt`.

### 4. Build results tables

```
python scripts/eval/build_tables.py --preds_root runs/preds --out_dir runs/tables \
  --configs G1 G2 E1 E2 V1 --seeds 42 43 44
```

Outputs per-seed metrics, protein-clustered bootstrap CIs, and paired tests.

## Experiments

- `scripts/experiments/verify_antisymmetry.py`            forward/reverse violation on independent inputs
- `scripts/experiments/verify_antisymmetry_symmetric.py`  same, for the symmetric variant
- `scripts/experiments/af_vs_pdb_control.py`              AlphaFold vs experimental-PDB structure-source control
- `scripts/eval/measure_params.py`                        trainable/total params, memory, throughput
- `scripts/eval/audit_claims.py`                          regenerate all numeric claims from prediction files

## Model selection

The final configuration is selected on the held-out Megascale test split, then
evaluated unchanged on all external benchmarks.
