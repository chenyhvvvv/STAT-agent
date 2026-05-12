---
name: deconvolution-cell2location
title: Bayesian Cell Type Deconvolution (Cell2location)
slug: deconvolution-cell2location
description: Reference-based Bayesian deconvolution of spot-level spatial transcriptomics using Cell2location. Two-stage model that first learns cell type expression signatures from scRNA-seq reference, then maps them to spatial spots. Provides uncertainty estimates. GPU recommended.

filter_requirements:
  modalities: [gene]
  data_levels: [spot]

prerequisites:
  - Annotated single-cell reference dataset path (.h5ad file)
  - Cell type column name in reference (default celltype)
  - GPU recommended (CPU works but slow)

default_skill: false
---

# Bayesian Cell Type Deconvolution (Cell2location)

Reference-based **Bayesian deconvolution** using **Cell2location**. A two-stage probabilistic model:
1. **Reference model**: Learn cell type expression signatures from scRNA-seq
2. **Spatial model**: Map signatures to spatial spots, estimating cell abundance

Provides **uncertainty estimates** (confidence intervals) for each cell type proportion.

**Output**:
- `adata.obsm['deconv_weights']`: Proportions (n_spots x n_celltypes)
- `adata.obs['celltype']`: Dominant cell type per spot

---

## Workflow

### Step 1: Load and Prepare Data

```python
import scanpy as sc
import pandas as pd
import numpy as np

print("=" * 60)
print("STEP 1: Load and Prepare Data")
print("=" * 60)

# Load spatial data
slice_id = 0  # <-- SET TARGET SLICE
slice_obj = session.get_slice(slice_id)
ad_spatial = slice_obj.adata.copy()

if not slice_obj.is_spot_level:
    raise ValueError("Cell2location requires spot-level data (Visium)")

# Load reference
ad_sc = sc.read_h5ad('reference_scRNA.h5ad')  # <-- SET REFERENCE PATH

# IMPORTANT: Cell type column
celltype_key = 'celltype'  # <-- SET CELL TYPE COLUMN

assert celltype_key in ad_sc.obs.columns, (
    f"Column '{celltype_key}' not found in reference."
)

# Ensure unique names
ad_spatial.var_names_make_unique()
ad_sc.var_names_make_unique()

# Common genes
common_genes = list(set(ad_spatial.var_names) & set(ad_sc.var_names))
assert len(common_genes) >= 100, f"Only {len(common_genes)} common genes"

print(f"  Spatial: {ad_spatial.n_obs} spots, {ad_spatial.n_vars} genes")
print(f"  Reference: {ad_sc.n_obs} cells, {ad_sc.n_vars} genes")
print(f"  Cell types: {ad_sc.obs[celltype_key].nunique()}")
print(f"  Common genes: {len(common_genes)}")

# Subset to common genes
ad_spatial = ad_spatial[:, common_genes].copy()
ad_sc = ad_sc[:, common_genes].copy()
```

### Step 2: Reference Model — Learn Signatures

```python
print("\n" + "=" * 60)
print("STEP 2: Reference Model (learn cell type signatures)")
print("=" * 60)

import cell2location
from cell2location.models import RegressionModel

# Setup reference model
RegressionModel.setup_anndata(ad_sc, labels_key=celltype_key)

# Train reference model
ref_model = RegressionModel(ad_sc)
ref_model.train(max_epochs=250, accelerator="auto")  # Set accelerator=False if no GPU

print(f"  Reference model trained (250 epochs)")

# Export learned signatures
ad_sc = ref_model.export_posterior(
    ad_sc, sample_kwargs={'num_samples': 1000, 'batch_size': 2500}
)

# Cell type signatures
inf_aver = ad_sc.varm['means_per_cluster_mu_fg'][[
    f'means_per_cluster_mu_fg_{c}' for c in ad_sc.uns['_scvi']['extra_categoricals']['mappings'][celltype_key]
]].copy()
inf_aver.columns = ad_sc.uns['_scvi']['extra_categoricals']['mappings'][celltype_key]

print(f"  Exported {inf_aver.shape[1]} cell type signatures")
```

### Step 3: Spatial Model — Map to Spots

```python
print("\n" + "=" * 60)
print("STEP 3: Spatial Model (map to spots)")
print("=" * 60)

from cell2location.models import Cell2location

# Setup spatial model
Cell2location.setup_anndata(ad_spatial)

# Estimated cells per spot (adjust for tissue density)
n_cells_per_spot = 30  # <-- ADJUST: ~5 for sparse, ~30 for dense tissue

spatial_model = Cell2location(
    ad_spatial,
    cell_state_df=inf_aver,
    N_cells_per_location=n_cells_per_spot,
    detection_alpha=20,
)

spatial_model.train(
    max_epochs=30000,
    batch_size=None,
    train_size=1,
    accelerator="auto",  # Set False if no GPU
)

print(f"  Spatial model trained")

# Export results
ad_spatial = spatial_model.export_posterior(
    ad_spatial,
    sample_kwargs={'num_samples': 1000, 'batch_size': ad_spatial.n_obs},
)
```

### Step 4: Store Results

```python
print("\n" + "=" * 60)
print("STEP 4: Store Results")
print("=" * 60)

# Extract proportions
abundance_key = 'means_cell_abundance_w_sf'
if abundance_key in ad_spatial.obsm:
    raw_weights = ad_spatial.obsm[abundance_key].copy()
else:
    # Fallback: q05_cell_abundance_w_sf
    abundance_key = 'q05_cell_abundance_w_sf'
    raw_weights = ad_spatial.obsm[abundance_key].copy()

# Normalize to proportions
row_sums = raw_weights.sum(axis=1)
proportions = raw_weights.div(row_sums, axis=0)

# Clean column names (remove prefix)
proportions.columns = [c.replace(f'{abundance_key}_', '') for c in proportions.columns]

# Store results
slice_obj.adata.obsm['deconv_weights'] = proportions
slice_obj.adata.uns['has_deconv_weights'] = True

# Dominant cell type
dominant = proportions.idxmax(axis=1)
slice_obj.adata.obs['celltype'] = dominant.values

print(f"  Stored deconv_weights ({proportions.shape})")
print(f"  Stored dominant celltype in adata.obs['celltype']")

# Summary
print(f"\nCell type proportions (mean across spots):")
mean_props = proportions.mean(axis=0).sort_values(ascending=False)
for ct, prop in mean_props.items():
    print(f"  {ct}: {prop:.3f}")
```

---

## Parameter Guide

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `celltype_key` | `'celltype'` | Any ref obs column | Cell type column in reference |
| `n_cells_per_spot` | 30 | 5-50 | Expected cells per spot |
| `ref_epochs` | 250 | 100-500 | Reference model training epochs |
| `spatial_epochs` | 30000 | 10000-50000 | Spatial model training epochs |
| `accelerator` | True | True/False | GPU acceleration |

## Notes

- Cell2location is a two-stage Bayesian model — slower than FlashDeconv/RCTD but provides uncertainty estimates.
- GPU strongly recommended. CPU training can take hours.
- Same output format as RCTD/FlashDeconv: `adata.obsm['deconv_weights']` and `adata.obs['celltype']`.
- Reference should have well-annotated, diverse cell types. More cells per type = better signature estimation.
- The `n_cells_per_spot` parameter should match tissue density (~5 for sparse, ~30 for Visium).
