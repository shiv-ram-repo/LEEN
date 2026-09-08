# protstab_data

Standalone PyTorch data-loading and feature-extraction library for **prot**ein
**stab**ility prediction. Developed as part of the LEEN project; installable and
usable independently, with no dependency on any external stability-prediction
package.

## What you get

- PDB parsing into per-chain backbone-coordinate dictionaries
- ProteinMPNN-style structural featurization and ESM-aware batch conversion
- Dataset classes for all eleven stability benchmarks:
  - `MegaScaleDataset` (Tsuboyama 2023)
  - `FireProtDataset`
  - `ddgBenchDataset` (Ssym direct, Ssym inverse, S669)
  - `ddgGeoDataset` (S461, S783, S8754, S2648, S571, S4346)
  - `DomainomeDataset` (Human Domainome)
  - `MegaScaleTestDatasets` (all eleven bundled)

## Installation

```bash
pip install torch numpy pandas tqdm biopython fair-esm
pip install lmdb atom3d   # only if you use LMDB-backed datasets
cd protstab_data && pip install -e .
```

## Expected data layout

Pass a `data_root` containing:

```
data_root/data/dataset/
├── megascale/
│   ├── Tsuboyama2023_Dataset2_Dataset3_20230416.csv
│   ├── mmseq_mut_search_0.25.m8
│   ├── mega_splits.pkl
│   └── AlphaFold_model_PDBs/<wt_name>.pdb
├── fireprot/fireprot_upload/{csvs,pdbs}/
├── ssym/{pdb, ssym-5fold_clean_dir.csv, ssym-5fold_clean_inv.csv}
├── S669/{pdb, s669_clean_dir.csv}
└── geostab_data/
    ├── ddG_cleaned/{S461,S783,S8754,S2648}{,.csv}
    └── dTm_cleaned/{S571,S4346}{,.csv}
```

A `parsed_structure.json` cache is generated automatically on first run.

## Quick start

```python
from torch.utils.data import DataLoader
from protstab_data import Alphabet, Featurizer, MegaScaleDataset, MegaScaleTestDatasets

DATA_ROOT = "/path/to/data/root"

alphabet   = Alphabet(name="esm")
featurizer = Featurizer(alphabet)

train_ds = MegaScaleDataset(data_root=DATA_ROOT, split="train")
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                          num_workers=4, collate_fn=featurizer)

test_collection = MegaScaleTestDatasets(data_root=DATA_ROOT)
for name, dataset in test_collection.iter_named():
    loader = DataLoader(dataset, batch_size=1, collate_fn=featurizer)
    print(f"{name}: {len(dataset)} proteins")
```

Each `__getitem__` returns one wild-type protein dict with all its mutations
bundled; use `Featurizer` as the `collate_fn`. A batch contains backbone
coordinates (`X`), sequence indices (`S`), masks, ESM `tokens`, the mutated
positions (`mut_ids`), ground-truth `ddG`, and `append_tensors` holding the
per-mutation wild-type/mutant one-hot pair.

Given a model output `phi` of shape `(L, 20)`, the predicted ΔΔG for a mutation
`wt -> mut` at position `i` is `phi[i, mut_idx] - phi[i, wt_idx]`.

## Module map

```
protstab_data/
├── __init__.py       public API
├── pdb_parser.py     PDB parsing to per-chain dicts + JSON cache
├── featurizer.py     tied_featurize, get_pdb, Alphabet, CoordBatchConverter, Featurizer
├── lmdb_dataset.py   optional LMDB-backed dataset
└── datasets/         megascale, fireprot, ddgbench, ddggeo, domainome, combined
```

## Dependencies

`torch`, `numpy`, `pandas`, `tqdm`, `biopython`, `fair-esm`. `lmdb` and `atom3d`
are only needed for `LMDBDataset`.
