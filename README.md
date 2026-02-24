# TCGA GBM scRNA-seq Analysis

Single-cell RNA sequencing (scRNA-seq) analysis of **Glioblastoma Multiforme (GBM)** samples — a subset from the [CPTAC-3 (Clinical Proteomic Tumor Analysis Consortium)](https://portal.gdc.cancer.gov/projects/CPTAC-3) project on the GDC Data Portal — using probabilistic deep learning with **scvi-tools**.

This project performs batch-corrected integration of 17 GBM patient samples, Bayesian differential expression, and unsupervised cell type annotation — providing a reproducible framework for dissecting the GBM tumor microenvironment at single-cell resolution.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data](#data)
- [Analysis Pipeline](#analysis-pipeline)
- [Getting Started](#getting-started)
- [Hardware Acceleration](#hardware-acceleration)
- [Requirements](#requirements)
- [Results](#results)
- [License](#license)

---

## Overview

Glioblastoma is the most aggressive primary brain tumor, characterized by substantial inter- and intra-tumoral heterogeneity. This project leverages scRNA-seq data from the CPTAC-3 project on GDC to:

1. **Construct an h5ad dataset** from raw GDC downloads (17 patient samples)
2. **Integrate across batches** using a Variational Autoencoder ([scVI](https://docs.scvi-tools.org/en/stable/)) to remove technical confounders while preserving biological variation
3. **Identify cell populations** via Leiden clustering on the batch-corrected latent space
4. **Perform differential expression** using scVI's Bayesian framework for calibrated uncertainty estimates
5. **Annotate cell types** with canonical GBM and brain cell markers (e.g., oligodendrocytes, astrocytes, tumor cells, immune cells)

---

## Project Structure

```
TCGA_GBM_scRNA-seq/
│
├── TCGA_GBM_Preprocessing.ipynb       # Data ingestion — builds h5ad from GDC downloads
├── scVI_GBM_analysis.ipynb            # Full analysis — QC, scVI integration, clustering, DE, annotation
├── model_training.py                  # Standalone scVI training script (CUDA)
│
├── gdc_sample_sheet.2026-02-24.tsv    # GDC sample sheet for reproducing the 17-case dataset
├── data/
│   └── gdc_extract/                   # Raw GDC scRNA-seq files (17 samples)
│       ├── <sample_uuid>/             # Individual patient sample directories
│       └── MANIFEST.txt               # GDC download manifest
│
├── scvi_model_gbm/                    # Saved scVI model checkpoint
├── requirements.txt                   # Python dependencies
│
└── README.md
```

---

## Data

**Source:** A GBM subset from the [CPTAC-3 project](https://portal.gdc.cancer.gov/projects/CPTAC-3) on the [GDC Data Portal](https://portal.gdc.cancer.gov/)

- **17 patient samples** of scRNA-seq data downloaded from GDC (CPTAC-3)
- Raw data is extracted and assembled into an [AnnData](https://anndata.readthedocs.io/) `.h5ad` object in `TCGA_GBM_Preprocessing.ipynb`
- The assembled dataset (`gbm_data.h5ad`) contains:
  - **Raw integer counts** in `adata.layers["counts"]` (required by scVI)
  - **SCT-normalized expression** in `adata.X`
  - **Batch labels** per patient in `adata.obs["batch"]`

> **Note:** The `.h5ad` data files and raw GDC downloads are excluded from version control due to their size (~2–3 GB each). Use the sample sheet and steps below to reproduce the dataset locally.

### Reproducing the Dataset

The included [`gdc_sample_sheet.2026-02-24.tsv`](gdc_sample_sheet.2026-02-24.tsv) contains the exact 17 cases used in this analysis. All samples are **primary GBM tumors** stored as Seurat `.loom` files.

<details>
<summary><b>17 CPTAC-3 Case IDs</b> (click to expand)</summary>

| Case ID | Sample ID | Tissue Type |
|------------|--------------|-------------|
| C3N-03188 | C3N-03188-02 | Tumor |
| C3N-03186 | C3N-03186-01 | Tumor |
| C3L-03405 | C3L-03405-01 | Tumor |
| C3N-03184 | C3N-03184-02 | Tumor |
| C3N-02190 | C3N-02190-01 | Tumor |
| C3N-01814 | C3N-01814-01 | Tumor |
| C3N-01816 | C3N-01816-01 | Tumor |
| C3N-02181 | C3N-02181-02 | Tumor |
| C3N-01815 | C3N-01815-01 | Tumor |
| C3N-01798 | C3N-01798-01 | Tumor |
| C3N-00662 | C3N-00662-03 | Tumor |
| C3N-02188 | C3N-02188-03 | Tumor |
| C3L-02705 | C3L-02705-71 | Tumor |
| C3N-02783 | C3N-02783-05 | Tumor |
| C3N-02769 | C3N-02769-02 | Tumor |
| C3L-03968 | C3L-03968-01 | Tumor |
| C3N-02784 | C3N-02784-01 | Tumor |

</details>

**To download the data from GDC:**

1. Go to the [GDC Data Portal](https://portal.gdc.cancer.gov/) and navigate to the **CPTAC-3** project
2. Filter for **Data Category:** Transcriptome Profiling → **Data Type:** Single Cell Analysis
3. Select the 17 cases listed above (or import the sample sheet directly)
4. Download the files using the GDC Data Transfer Tool or the portal's cart
5. Place the downloaded archive in the project root and run `TCGA_GBM_Preprocessing.ipynb` to assemble `gbm_data.h5ad`

---

## Analysis Pipeline

### Notebook 1 — `TCGA_GBM_Preprocessing.ipynb` (Data Preparation)

| Step | Description |
|------|-------------|
| 1 | Define paths and inspect the GDC tarball |
| 2 | Extract raw scRNA-seq files |
| 3 | Auto-detect file types (`.h5`, `.h5ad`, `.loom`) and load per-sample data |
| 4 | Assemble multi-sample AnnData and save as `gbm_data.h5ad` |

### Notebook 2 — `scVI_GBM_analysis.ipynb` (Integration & Annotation)

| Step | Description |
|------|-------------|
| 1 | Load data and setup environment |
| 2 | Quality control — mitochondrial/ribosomal gene detection, filtering |
| 3 | Highly variable gene (HVG) selection (`seurat_v3`, 4 000 genes, batch-aware) |
| 4 | scVI model setup and training (30-dim latent space, negative binomial likelihood) |
| 5 | Latent space extraction, neighbor graph, and UMAP visualization |
| 6 | Leiden clustering at multiple resolutions |
| 7 | Bayesian differential expression with scVI |
| 8 | Cell type annotation using canonical marker genes |
| 9 | Save annotated AnnData (`gbm_scvi_annotated.h5ad`) and trained model |

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- [Conda](https://docs.conda.io/) or [pip](https://pip.pypa.io/) for package management

### Installation

```bash
# Clone the repository
git clone https://github.com/jjevans25/TCGA_GBM_scRNA-seq.git
cd TCGA_GBM_scRNA-seq

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Reproducing the Analysis

1. **Prepare the data** — Run `TCGA_GBM_Preprocessing.ipynb` to build `gbm_data.h5ad` from raw GDC files
2. **Run the analysis** — Execute `scVI_GBM_analysis.ipynb` end-to-end, or use the standalone training script:

```bash
python model_training.py
```

---

## Hardware Acceleration

The standalone training script (`model_training.py`) is configured for **CUDA** (NVIDIA GPU). To use a different accelerator, update the `accelerator` parameter in the `model.train()` call (e.g., `"mps"` for Apple Silicon, `"cpu"` for CPU-only).

Training parameters:
- **Max epochs:** 400 (with early stopping, patience = 20)
- **Latent dimensions:** 30
- **Network depth:** 2 layers
- **Likelihood:** Negative binomial
- **Batch size:** 256

---

## Requirements

Core libraries and their roles:

| Library | Purpose |
|---------|---------|
| [scanpy](https://scanpy.readthedocs.io/) | Preprocessing, clustering, visualization |
| [scvi-tools](https://docs.scvi-tools.org/) | Probabilistic modeling & batch integration |
| [PyTorch](https://pytorch.org/) | Deep learning backend |
| [leidenalg](https://leidenalg.readthedocs.io/) | Community detection (clustering) |
| [pybiomart](https://github.com/jrderuiter/pybiomart) | Ensembl ID → gene symbol mapping |
| [loompy](https://loompy.org/) | Loom file format support |

See [`requirements.txt`](requirements.txt) for the full dependency list.

---

## Results

The final annotated dataset (`gbm_scvi_annotated.h5ad`) contains:

- **Batch-corrected UMAP embeddings** revealing biologically meaningful structure across all 17 patients
- **Leiden cluster assignments** at multiple resolutions
- **Bayesian differential expression** results per cluster
- **Cell type annotations** mapped via canonical markers for GBM-relevant populations (e.g., neoplastic cells, TAMs, oligodendrocytes, T cells, endothelial cells)

---

## License

This project is for academic and research purposes. CPTAC-3 data is publicly available through the GDC and subject to the [GDC Data Use Agreement](https://gdc.cancer.gov/access-data/data-access-processes-and-tools).
