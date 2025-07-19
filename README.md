# Efficient Latent-Topic Modeling Unifies Cell-Type Deconvolution and Domain Discvoery for Multi-Section Spatial Transcriptomics

FUSION is a fast, unified method for multi-section SRT data analysis, which leverages a matched single-cell RNA reference to unify deconvolution, domain detection, and cross-section integration. FUSION directly models each sequencing read as arising from a latent topic that captures the transcriptomic programme of a reference cell type. Aggregating the read-level topic probabilities for each spot forms cell-type compositional embeddings whose softmax normalization yields cell-type proportions. Clustering directly in this compositional embedding space delivers interpretable domain detection while automatically aligning homologous regions across sections.

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
## Dependency

#### Python core 
- python ≥ 3.9, numpy ≥ 1.24, pandas ≥ 2.0, scipy ≥ 1.10, h5py ≥ 3.10, scikit‑learn ≥ 1.3, tqdm ≥ 4.66, rpy2 ≥ 3.5, matplotlib ≥ 3.8, seaborn ≥ 0.13

#### Deep‑learning and Spatial‑omics
- pytorch = 2.2, torchvision = 0.17, torchaudio = 2.2, scanpy ≥ 1.9, anndata ≥ 0.10, umap‑learn = 0.5.5
  
#### R runtime
- r‑base ≥ 4.3.3, r‑essentials ≥ 4.2.3  *(ggplot2, tidyverse, …)*, Matrix ≥ 1.7‑0, devtools ≥ 2.4.5, IRIS = 1.0.1:```devtools::install_github("YingMa0107/IRIS")```



## 🚀 Quick Start

<summary><strong>1 · Clone&nbsp;&amp;&nbsp;install</strong></summary>

```bash
# clone the repo
git clone https://github.com/<your‑org>/FUSION.git
cd FUSION

# create the Conda + R environment
conda env create -n fusion
conda activate fusion

```

<summary><strong>2 · Running&nbsp;&amp;&nbsp;Testing</strong></summary>

Key Inputs:

1. sp_expr​​: A ​​spot-by-gene matrix​​ (spatial transcriptomics data in matrix/dataframe format).
2. sp_pos​​: A ​​2D spatial coordinate matrix​​ (spot locations in X-Y coordinates).
3. top_DEGs​​: A ​​list of differentially expressed genes​​ (cell-type marker genes).
4. Num_topic​​ (int): Number of spatial domains to infer.
5. Num_HVG​​ (int): Number of highly variable genes (HVGs) to include in training.
6. dim_embed​​ (int): Latent dimension for hierarchical factor modeling.

Marker Gene Options: 

1. top_marker_num​​ (int): Only use the ​​top n marker genes​​ per cell type from top_DEGs.
2. fixed_marker_list​​ (logical):

    FALSE → Use top top_marker_num genes per cell type.
    TRUE → Use all genes in top_DEGs.

For a quick start example, see the `tutorial/MOB.ipynb`
