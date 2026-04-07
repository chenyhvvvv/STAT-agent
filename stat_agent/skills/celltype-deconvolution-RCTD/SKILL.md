---
name: celltype-deconvolution
title: Cell Type Deconvolution (RCTD)
slug: celltype-deconvolution
description: Perform cell type deconvolution (or annotation on spot) on spatial transcriptomics data (Visium spots) using RCTD with a single-cell reference dataset. (Recommended for spot data)

filter_requirements:
  num_slices: 1
  modalities: [gene]
  data_levels: [spot]

prerequisites:
  - Annotated single-cell reference dataset path (.h5ad file)
  - Cell type column name in the reference dataset (default celltype)
  - Spatial data must contain raw UMI counts (not normalized)
default_skill: true
---

# Cell Type Deconvolution using RCTD

Perform reference-based cell type deconvolution on spatial transcriptomics data (Visium, etc.) using RCTD (Robust Cell Type Decomposition).

## Overview
**Output**:
- `adata.obsm['deconv_weights']`: DataFrame with shape (n_spots, n_celltypes)
  - Rows: Spots from spatial data
  - Columns: Cell types from reference
  - Values: Proportion (0-1) of each celltype in that spot
- `adata.obs['celltype']`: Virtual celltype column with dominant celltype per spot
- Some spots may be filtered out due to low UMI counts

## Requirements

### Input Data
- **Spatial data**: Visium (or similar spot-based) AnnData object
  - Required: `adata.obs['x', 'y']` coordinates
  - Required: `adata.X` gene expression counts (not normalized!)
  - Should be raw UMI counts for best results

- **Reference single-cell data**: scRNA-seq AnnData object
  - Required: `adata.obs['celltype']` annotations
  - Required: `adata.X` gene expression counts
  - Should have diverse cell types representing tissue

- **Common genes**: Must have overlapping gene names between spatial and reference
  - Script will handle: gene name matching, normalization, marker selection

## Workflow

### Step 1: Prepare Data
```python
import scanpy as sc
import pandas as pd
import numpy as np

# Load spatial data
# The slice id should be specified: e.g. slice 0/1... 
# Example: slice 0
slice = session.get_slice(0)
ad_spatial = slice.adata.copy()

# Verify it's spot-level data
if not slice.is_spot_level:
    raise ValueError("Deconvolution requires spot-level data (Visium)")

# Load reference single-cell data, change to the reference path
ad_sc = sc.read_h5ad('reference_scRNA.h5ad')

# Make sure gene/cell names are unique
ad_spatial.var_names_make_unique()
ad_sc.var_names_make_unique()
ad_spatial.obs_names_make_unique()
ad_sc.obs_names_make_unique()
```

### Step 2: Select the markers in the reference
```python
# Idenfied markers
ad_sc.raw = ad_sc.copy()
sc.pp.normalize_total(ad_sc,target_sum=2000)

sc.pp.highly_variable_genes(ad_sc, flavor='seurat_v3',n_top_genes=1000)
sc.tl.rank_genes_groups(ad_sc, groupby="celltype", method='wilcoxon')
markers_df = pd.DataFrame(ad_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
markers = list(np.unique(markers_df.melt().value.values))
markers = list(set(ad_sc.var.loc[ad_sc.var['highly_variable']==1].index)|set(markers))
d_sc.var.loc[ad_sc.var.index.isin(markers),'Marker'] = True
ad_sc.var['Marker'] = ad_sc.var['Marker'].fillna(False)

ad_sc.var['highly_variable'] = ad_sc.var['Marker']
sel_genes = ad_sc.var.index[ad_sc.var['Marker']]
# Only preserve these marker in spatial data
ad_spatial = ad_spatial[:,ad_spatial.var.index.isin(sel_genes)]
```

### Step 2: Run Deconvolution
```python
# Import RCTD components
from deconv_rctd import SpatialRNA, Reference, create_RCTD, run_RCTD

# Prepare spatial data for RCTD
counts_spatial = ad_spatial.to_df().T
coords = ad_spatial.obs[['x', 'y']]
nUMI_spatial = pd.DataFrame(np.array(ad_spatial.X.sum(-1)),
                            index=ad_spatial.obs.index)
puck = SpatialRNA(coords, counts_spatial, nUMI_spatial)

# Prepare reference data for RCTD
counts_ref = ad_sc.to_df().T
cell_types_ref = pd.DataFrame(ad_sc.obs['celltype'])
nUMI_ref = pd.DataFrame(ad_sc.to_df().T.sum(0))
reference = Reference(counts_ref, cell_types_ref, nUMI_ref)

# Run RCTD
myRCTD = create_RCTD(puck, reference, max_cores=20)
myRCTD = run_RCTD(myRCTD)

# IMPORTANT: Shutdown Ray to free resources
import ray
if ray.is_initialized():
    ray.shutdown()

# IMPORTANT: Reset numba JIT state after Ray shutdown.
# Ray's forked workers corrupt numba's dispatcher registry, causing
# "resolving callee type: CPUDispatcher" errors in subsequent numba-
# dependent code (e.g., pynndescent used by sc.pp.neighbors).
try:
    import numba.core.registry
    numba.core.registry.cpu_target.typing_context.refresh()
    numba.core.registry.cpu_target.target_context.refresh()
except Exception:
    pass
```

### Step 3: Update Session Data
⚠️ **CRITICAL**: Some spots are filtered out during RCTD!
```python
# Get filtered spots from RCTD results
filtered_spots = myRCTD["results"].index

# Update session adata to only include filtered spots
slice.adata = slice.adata[filtered_spots, :].copy()

# Add deconvolution weights to obsm
slice.adata.obsm["deconv_weights"] = myRCTD["results"]
slice.adata.uns['has_deconv_weights'] = True

# Update the dominant celltype
dominant_celltypes = slice.adata.obsm["deconv_weights"].idxmax(axis=1)
slice.adata.obs['celltype'] = dominant_celltypes
```

## Outputs

After successful deconvolution:

### In adata.obsm
- **`deconv_weights`**: Shape (n_spots, n_celltypes)
  - Contains proportions (0-1) for each celltype in each spot
  - Access specific celltype: `adata.obsm['deconv_weights']['Celltype_Name']`

### In adata.obs
- **`celltype` (virtual)**: Dominant celltype per spot
  - Automatically created from highest proportion in deconv_weights
  - Useful for visualization and analysis


