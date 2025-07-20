# Efficient Latent-Topic Modeling Unifies Cell-Type Deconvolution and Domain Discvoery for Multi-Section Spatial Transcriptomics

FUSION is a fast method for multi‑section SRT that, with a matched scRNA‑seq reference, performs cell‑type deconvolution, spatial‑domain detection, and cross‑section alignment in a single probabilistic framework. Each read is assigned to a latent topic representing a reference cell type; spot‑level topic aggregates yield cell‑type proportions, and clustering these proportions reveals coherent spatial domains shared across sections.

## 📂 Repository Layout
```text
FUSION/
├── main_ref.py            # training / inference entry point
├── dataprocess.py         # preprocessing & gene filtering
├── r_batch.py             # WGAN batch‑effect removal
├── R_initialization.py    # R helpers called via rpy2
├── utils.py               # misc. utility functions
├── environment.yml        # reproducible Conda + R environment
├── LICENSE
├── README.md              # ← you are here
│
├── notebooks/             # interactive demos & benchmarks
│   ├── DLPFC_all.ipynb        # end‑to‑end run on 12 DLPFC slices
│   └── DLPFC_batch_remove.ipynb
│
├── dataset/               # toy data to get you started
│   ├── SC_data/               # reference scRNA‑seq (AnnData)
│   └── SRT_data/              # Visium slides (.h5ad)
│
└── tests/                 # unit / smoke tests
    └── test_inference.py
```
## 🛠️ Dependencies

#### Python core 
- python ≥ 3.9, numpy ≥ 1.24, pandas ≥ 2.0, scipy ≥ 1.10, h5py ≥ 3.10, scikit‑learn ≥ 1.3, tqdm ≥ 4.66, rpy2 ≥ 3.5, matplotlib ≥ 3.8, seaborn ≥ 0.13

#### Deep‑learning and Spatial‑omics
- pytorch = 2.2, torchvision = 0.17, torchaudio = 2.2, scanpy ≥ 1.9, anndata ≥ 0.10, umap‑learn = 0.5.5
  
#### R runtime
- r‑base ≥ 4.3.3, r‑essentials ≥ 4.2.3  *(ggplot2, tidyverse, …)*, Matrix ≥ 1.7‑0, devtools ≥ 2.4.5, IRIS = 1.0.1:```devtools::install_github("YingMa0107/IRIS")```



## 🏃‍♂️ Using FUSION – step‑by‑step

<details>
<summary><strong>1 · Clone&nbsp;&amp;&nbsp;install</strong></summary>

```bash
# clone the repo
git clone https://github.com/<your‑org>/FUSION.git
cd FUSION

# create the Conda + R environment
conda env create -n fusion
conda activate fusion

```
</details>

<details>
<summary><strong>2 · Running&nbsp;&amp;&nbsp;Testing</strong></summary>

Before running FUSION, prepare the inputs below:

---

| Object | Required fields | Example path |
|--------|-----------------|--------------|
| **SRT slides** | `AnnData` (`.h5ad`) with <br>• `.X` = raw spot‑by‑gene counts<br>• `adata.obsm["spatial"]` = `[[x, y], …]` | `dataset/SRT_data/151507_adata.h5ad` |
| **scRNA‑seq reference** | `AnnData` with `obs["cellType"]` labels | `dataset/SC_data/scref_adata.h5ad` |

Group slides that belong to the **same patient / condition** into an inner list;  
collect those inner lists into `adata_list`, e.g.

```python
import scanpy as sc
# three patients, four slides each
adata_list = [
    [sc.read_h5ad(f"dataset/SRT_data/{sid}_adata.h5ad")
     for sid in ("151507","151508","151509","151510")],
    [sc.read_h5ad(f"dataset/SRT_data/{sid}_adata.h5ad")
     for sid in ("151669","151670","151671","151672")],
    [sc.read_h5ad(f"dataset/SRT_data/{sid}_adata.h5ad")
     for sid in ("151673","151674","151675","151676")]
]
sc_adata = sc.read_h5ad("dataset/SC_data/scref_adata.h5ad")

from R_initialization import FUSION_Init    
FUSION_Init(adata_list, sc_adata, domain_size)

from main_ref import FUSION_preprocess, FUSION_main
log_fc_cut = 1.5         # log‑fold‑change threshold for marker filtering
FUSION_preprocess(adata_list, log_fc_cut)

out, emb = FUSION_main(adata_list, embed_dim=64, domain_size=7)
```

For an illustrative example on DLPFC, see the Jupyter notebook: `Jupyter notebook` for details.
</details>
