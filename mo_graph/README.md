# `mo_graph`: dual-encoder reranking (MIST spectrum encoder × molecular GNN)

This is Mohammed's part of the CASMI26 project: proposal Architecture 2, "spectrum–graph matching". MIST is the
spectrum encoder, which the proposal said would be "tested if feasible".

The whole pipeline is in [`dual_encoder_reranker.ipynb`](dual_encoder_reranker.ipynb), on the full Kaggle data. Figures
are in `figures/` and tables in `results/`.

## What it does

The task follows Enveda CASMI 2026: rank candidate structures for each molecule, scored by MRR@25 on the InChIKey first
block. The candidates are every catalog structure (all 275,810 training structures) within 10 ppm of the neutral mass
implied by the precursor m/z and adduct. There is no fingerprint shortlist. Each candidate is scored by three parts:

1. **MIST-style spectrum encoder** (Goldman et al., 2023). Peaks are annotated with subformulae of *the candidate's*
   formula, so no ground-truth formula is needed at inference. A transformer encodes the fragment and neutral-loss
   formula tokens with a pairwise formula-difference attention bias. An auxiliary head predicts the 1,024-bit Morgan
   fingerprint, the same target as DeepSets.
2. **GINE graph encoder** over the RDKit atom and bond graph of the candidate SMILES.
3. **Matching MLP** over `[z_s ⊙ z_g, |z_s − z_g|, cos]`, plus mass error, explained intensity, the MIST fingerprint
   log-likelihood and a learnable Gaussian mass-error prior. It is trained with a listwise softmax loss against 15 hard
   negatives (same-formula isomers and other formulas from the same mass window).

When a molecule has several spectra, the per-spectrum log-softmax scores are averaged. This is the proposal's
multi-spectrum question.

The released MIST checkpoint (Zenodo 8316682) is not used because:
- it was trained on NPLIB1 for `[M+H]+` only;
- it needs Python 3.8 / torch 1.9 / Lightning 1.6 on Linux;
- it predicts 4,096-bit fingerprints.

Instead, MIST's two key ideas are re-implemented and trained on our split.

## Consistency with the team code

- **Held-out structures** never enter training, not even as negatives:
  - Milton's validation structures, so his 8,068-spectrum benchmark stays comparable;
  - Antoine's validation hash (`streamingDataset.get_split`);
  - all 250 `enveda-np-examples` natural products, which Kaggle calls the library closest to the hidden test set.
- Adducts and neutral mass follow Antoine's `config.ADDUCTS` (the generic parser is asserted to reproduce it). The
  fingerprint uses `config.FINGERPRINT_SIZE` / `FINGERPRINT_RADIUS`, and the data path is `config.DATA_PATH`.
- Metrics are Top-1, Top-5 and MRR@25, with Milton's tie-averaged expected rank. Missing candidates score zero.
- **Baselines run on the same pools:**
  - mass only;
  - Milton's DeepSets checkpoint;
  - the MIST fingerprint head alone;
  - an ablation with the GNN removed, everything else identical.

## Run it

```powershell
python -m venv venv
venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
venv\Scripts\pip install rdkit pyarrow pandas numpy matplotlib scikit-learn ipykernel
```

Data, all git-ignored:
- `data/enveda-CASMI26-molecule-id-mass-spectra/`: `train.parquet`, `test.parquet` and `sample_submission.csv` from
  Kaggle.
- `data/recap/`: Milton's recap zip (`data/*` and `models/`).

Run the notebook from `mo_graph/`. Timings on an RTX 3050 Ti (4 GB):
- **Full run:** about 75 minutes. Featurising the catalog takes about 3 minutes the first time and is then cached.
- **From checkpoints:** about 6 minutes. Existing checkpoints in `mo_graph/checkpoints/` are reused unless
  `RETRAIN=1`.
- **Smoke test** (`QUICK=1`): about 12 minutes.

Progress goes to `mo_graph/train_log.txt`. On Windows, keep the repo and venv at a short path, because RDKit's DLLs
fail to load past 260 characters.

## Results

Setup:
- seed 42, 4 epochs;
- 120,000 training spectra, one per structure;
- full catalog of 275,810 structures;
- 95% CIs from 1,000 bootstrap resamples of whole structures or molecules.

**Milton's benchmark:** 8,068 validation spectra, scored per spectrum, candidate recall 94.9%.

| Method | Top-1 | Top-5 | MRR@25 |
|---|---:|---:|---:|
| Mass only | 0.218 | 0.492 | 0.344 |
| DeepSets FP (Milton's checkpoint) | 0.161 | 0.410 | 0.276 |
| MIST FP | 0.174 | 0.465 | 0.305 |
| Dual, no GNN | 0.315 | 0.651 | 0.463 |
| **Dual MIST + GNN** | **0.415** | **0.753** | **0.560** |

**Natural products on the test instrument:** `enveda-np-examples`, 250 molecules and 1,184 timsTOF spectra, scored
per molecule.

| Method | MRR@25, one spectrum | MRR@25, aggregated | Top-1, aggregated |
|---|---:|---:|---:|
| Mass only | 0.380 | 0.392 | 0.221 |
| DeepSets FP | 0.315 | 0.314 | 0.164 |
| MIST FP | 0.344 | 0.345 | 0.184 |
| Dual, no GNN | 0.508 | 0.522 | 0.328 |
| **Dual MIST + GNN** | **0.590** | **0.627** | **0.452** |

Paired MRR@25 differences with 95% CIs:

| Comparison | Milton's benchmark | Natural products |
|---|---|---|
| Dual vs mass only | +0.216 [0.205, 0.228] | +0.235 [0.191, 0.277] |
| Dual vs its no-GNN ablation | +0.096 [0.086, 0.107] | +0.104 [0.061, 0.146] |
| MIST FP vs DeepSets FP | +0.029 [0.022, 0.036] | |
| Aggregated vs one spectrum (dual) | | +0.036 [0.019, 0.054] |

Mass-only and DeepSets reproduce Milton's full-catalog numbers (0.336 / 0.266).

The GNN matters more as isomer ambiguity grows. With more than 100 same-formula candidates, MRR@25 is:
- 0.02 for mass alone;
- 0.13 for the dual model without the GNN;
- 0.31 for the dual model with the GNN.

**Cost:** 4.7M parameters, 37 min training, 1.4 GiB peak GPU memory, and about 10 ms per spectrum.

**Figures** (`figures/`):
- `00` architecture
- `01` subformula annotation
- `02` candidate pools
- `03` training curves
- `04` main results
- `05` breakdown by isomer count and library
- `06` example ranking
- `07` natural products, one spectrum vs aggregated

The earlier pilot on the ~50k-structure sample catalog is kept in `results/pilot_sample_catalog/`.

### Kaggle

`results/submission.csv` follows `sample_submission.csv`: 400 `molecule_id`s with 25 SMILES each, built from
`test.parquet`. CASMI26 is a **code competition**, so only a `submission.csv` produced by a Kaggle notebook running
offline is scored. Competing would require two steps:
1. upload the catalog cache and checkpoints as a Kaggle Dataset;
2. run this inference code in a Kaggle notebook.

Retrieval also cannot recover Kaggle's class-3 (novel) structures.

## Literature

- **MIST:** Goldman et al., *Nat. Mach. Intell.* 5, 965–979 (2023). Formula-annotated peak transformer; the spectrum
  encoder here.
- **JESTR:** Kalia et al., *Bioinformatics* btaf354 (2025). Joint embedding of spectra and molecular graphs (GNN),
  trained against candidate sets. This is the closest prior work.
- **MassSpecGym:** Bushuiev et al., NeurIPS Datasets & Benchmarks (2024). Retrieval protocol; MIST is its strongest
  retrieval baseline.
- **CSI:FingerID:** Dührkop et al., *PNAS* 112, 12580 (2015). Fingerprint retrieval.
- **GIN / GINE:** Xu et al., ICLR (2019); Hu et al., ICLR (2020).
- **MPNN:** Gilmer et al., ICML (2017).
- **DeepSets:** Zaheer et al., NeurIPS (2017).
