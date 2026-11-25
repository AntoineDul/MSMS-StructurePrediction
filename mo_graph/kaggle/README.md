# Kaggle submission (offline code competition)

CASMI26 only scores a `submission.csv` written by a Kaggle notebook. Kaggle re-runs that notebook offline on the
hidden test set (≤ 9 h, internet off). [`casmi26_dual_reranker_inference.ipynb`](casmi26_dual_reranker_inference.ipynb)
is that notebook. It needs only numpy, pandas, pyarrow and torch, all preinstalled on Kaggle.

## 1. Build the bundle

Run `../dual_encoder_reranker.ipynb`. Its section 12 writes `bundle/` (git-ignored, ~165 MB):

| File | Content |
|---|---|
| `dualrerank.py` | model, chemistry and scoring code (same file as `../dualrerank.py`) |
| `catalog_arrays.npz`, `catalog_meta.parquet` | 725,209 candidate structures (training + COCONUT): graphs, fingerprints, masses, SMILES |
| `dual_mist_gnn.pt` | trained dual-encoder weights |
| `config.json` | model size, adduct vocabulary, tolerances |

## 2. Check it locally

Run the notebook from this folder; it falls back to `bundle/` and `../../data/...`. It writes `submission.csv` here.

## 3. Upload and submit

These steps use your own Kaggle account. Do them yourself, or ask Claude to run them once your token is set up.

```powershell
pip install kaggle
```

Put your token in the `KAGGLE_API_TOKEN` environment variable or in `~/.kaggle/access_token`. Then:

```powershell
kaggle datasets init -p mo_graph/kaggle/bundle
```

```powershell
kaggle datasets create -p mo_graph/kaggle/bundle
```

Before `create`, edit `bundle/dataset-metadata.json`:
- set `"id": "<your-username>/casmi26-dual-reranker"`;
- set a title.

The dataset is private unless you pass `--public`.

```powershell
kaggle kernels push -p mo_graph/kaggle
```

Before pushing, replace `<your-username>` in `kernel-metadata.json`. That file attaches:
- the dataset;
- the competition data;
- a GPU;
- internet off.

When the notebook run finishes on Kaggle, open it and click **Submit to competition**. Alternatively:

```powershell
kaggle competitions submit -c enveda-CASMI26-molecule-id-mass-spectra -k <your-username>/casmi26-dual-reranker-inference -f submission.csv -v 1 -m "dual MIST+GNN reranker"
```

To update an existing dataset after re-exporting the bundle, run `kaggle datasets version -p mo_graph/kaggle/bundle -m "<note>"`.
Then push the notebook again and submit that version number with `-v`.
