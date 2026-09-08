# LEEN — Local Equivariant Energy Network

ΔΔG protein stability prediction via an ESM-gated E(n)-equivariant energy
network with an energy-difference readout (ΔΔG = E(mut) − E(wt)), evaluated on
eleven stability benchmarks.

This repository accompanies the manuscript *"LEEN: Local Equivariant Energy
Networks with Evolution-Guided Structural Reasoning for Protein Stability
Prediction."* All tables and figures in the paper are regenerated from the
per-sample predictions released under `results/`.

---

## 1. Installation

```bash
# clone, then from the repo root:
pip install -e .                     # installs the leen package
pip install -e protstab_data/        # installs the bundled data package (see §3)
pip install -r requirements.txt      # remaining Python dependencies
```

External tool (only needed to *reproduce* the sequence-similarity filtering, §4):

```bash
conda install -c bioconda mmseqs2    # or see https://github.com/soedinglab/MMseqs2
```

Dependencies: PyTorch ≥2.0, NumPy, SciPy, pandas, tqdm, PyYAML, fair-esm.
Figures additionally require matplotlib. Tested with Python 3.11 on a single
NVIDIA A6000 (CUDA 12).

---

## 2. Repository layout

```
leen/                     Core model + data code
  models/                 leen_v2.py (LEENv2), egnn.py, gvp_encoder.py, energy_model.py
  data/                   adapter_v2.py (local-graph builder), featurizer.py,
                          adapter_v2_symmetric.py / _masked.py (anti-symmetry variants)
  losses/losses.py        BalancedMSE (BMC), OOD-margin, LEENLoss
protstab_data/            Bundled dataset/featurization package (see §3)
scripts/
  preprocess/             ESM-2 precompute, MMseqs2 filter documentation
  train/                  train_leen.py (+ concat / esm_mlp / symmetric / masked)
  eval/                   eval_persample.py, build_tables.py, measure_params.py, audit_claims.py
  experiments/            verify_antisymmetry*.py, af_vs_pdb_control.py
configs/default.yaml      All hyperparameters (architecture + training)
results/                  Released per-sample predictions + regenerated tables
```

---

## 3. The `protstab_data` package

`protstab_data/` is our own standalone data-loading package (named for
**prot**ein **stab**ility **data**), developed as part of this project with no
dependency on any external stability-prediction package. It provides PDB
parsing, ProteinMPNN-style structural featurization, and dataset classes for all
eleven benchmarks (Megascale, FireProt-HF, Ssym direct/inverse, S669, S461,
S783, S8754, S2648, S571, S4346). It is bundled here and installed with
`pip install -e protstab_data/`.

---

## 4. Data preparation

### 4.1 Download the datasets

| Dataset | Source |
|---|---|
| Megascale (Tsuboyama 2023) | https://doi.org/10.1038/s41586-023-06328-6 (Zenodo release linked therein) |
| FireProtDB / FireProt-HF | https://loschmidt.chemi.muni.cz/fireprotdb/ |
| Ssym (direct + inverse), S669 | Pucci 2018; Pancotti 2022 |
| S461, S783, S2648, S8754, S571, S4346 | ddgGeo benchmark collections (see manuscript refs) |
| Protein structures | Experimental: RCSB PDB (https://www.rcsb.org). Training: AlphaFold2 models from the Megascale release. |

Arrange them under a single `--data_root`:

```
data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv
data/dataset/megascale/mmseq_mut_search_0.25.m8          # MMseqs2 result (see §4.2)
data/dataset/megascale/mega_splits.pkl
data/dataset/megascale/AlphaFold_model_PDBs/<wt_name>.pdb
data/dataset/<benchmark>/...                             # one dir per benchmark
```

### 4.2 Sequence-similarity filtering (MMseqs2)

All benchmark proteins are held non-redundant with the Megascale training set at
**0.25 sequence identity**. The filtering result is stored as
`mmseq_mut_search_0.25.m8`. To regenerate it, build a target DB from the training
sequences and a query DB from the benchmark sequences, then:

```bash
mmseqs easy-search benchmark.fasta train.fasta mmseq_mut_search_0.25.m8 tmp \
  --min-seq-id 0.25 -c 0.5 --cov-mode 0 -s 7.5
```

`scripts/preprocess/document_mmseqs_filter.py` audits the `.m8` and writes the
list of removed proteins/rows (28,133 rows / 94 benchmark-overlapping proteins
removed at 0.25). The retained/removed lists are released under `results/`.

### 4.3 Precompute frozen ESM-2 embeddings

```bash
python scripts/preprocess/precompute_esm.py --data_root DATA --output esm_cache_full.pt --fp16
```

---

## 5. Training

Main model (E2 = ESM-gated EGNN + cross-attention + OOD-margin loss):

```bash
python scripts/train/train_leen.py --data_root DATA --esm_cache esm_cache_full.pt \
  --output_dir runs/E2 --seed 42
```

Configuration flags (map to the paper's ablation table):

| Flag | Effect | Config |
|---|---|---|
| `--encoder {egnn,gvp}` | geometric encoder | egnn = E1/E2, gvp = V1 |
| `--no_cross_attn` | gating only, no cross-attention | E1 |
| `--no_ood` | disable OOD-margin consistency loss | no-OOD ablation |
| `--esm_dropout FLOAT` | ESM-dropout rate (default 0.25; 0 disables) | no-ESM-drop ablation |
| `--head_type {energy,scalar}` | energy-difference (default) or direct scalar regressor | scalar-head ablation |

Geometry-only (G1) uses `train_leen.py` with ESM disabled; concatenation (G2)
uses `train_concat.py`; the structure-free baseline uses `train_esm_mlp.py`.
Exact-anti-symmetry variants: `precompute_esm_symmetric.py` then
`train_symmetric.py`. Key training settings (configs/default.yaml):
batch_size 32, max_epochs 200, early-stopping patience 20 on validation
Spearman, AdamW lr 1e-4, energy_hidden_dim 256, seeds 42/43/44.

---

## 6. Evaluation and results

```bash
# per-sample predictions for a checkpoint
python scripts/eval/eval_persample.py --ckpt runs/E2/seed_42/best.pt --family v2 \
  --esm_cache esm_cache_full.pt --data_root DATA --out_dir results/per_sample_predictions/E2_seed42
# (add --symmetric --sym_esm_cache sym_esm_cache_full.pt for symmetric checkpoints)

# regenerate every table in the paper from the per-sample predictions
python scripts/eval/build_metrics_tables.py --preds_root results/per_sample_predictions --out_dir results/paper_tables
```

The `eval_persample.py` model builder reads `head_type` (and all other config)
from the checkpoint, so energy and scalar checkpoints are evaluated with the
correct readout automatically.

`results/per_sample_predictions/<CONFIG>_seed<N>/<benchmark>.csv` contains the
released predictions (columns: `protein_idx, mut_index, pred, true`) for every
configuration, seed, and benchmark. `results/paper_tables/all_metrics.csv` is the
single source of truth for all reported numbers.

---

## 7. Controls and diagnostics

- `scripts/experiments/verify_antisymmetry.py` — forward/reverse violation under
  independently constructed inputs (primary E2).
- `scripts/experiments/verify_antisymmetry_symmetric.py` — same for the
  symmetrized variant (violation ≈ 1e-7, i.e. exact to float precision).
- `scripts/experiments/af_vs_pdb_control.py` — AlphaFold-vs-experimental-PDB
  structure-source control (ΔρE2 ≈ −0.003 on S669 and S461).
- `scripts/eval/measure_params.py` — trainable/total parameters, peak memory,
  throughput.

---

## 8. Citation

If you use this code, please cite the LEEN manuscript.
