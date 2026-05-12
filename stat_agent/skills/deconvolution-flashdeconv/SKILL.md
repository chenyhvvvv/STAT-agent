---
name: deconvolution-flashdeconv
title: Fast Spot Deconvolution (FlashDeconv)
slug: deconvolution-flashdeconv
description: Ultra-fast reference-based cell type deconvolution for spot-level spatial data using FlashDeconv. O(N) complexity via random sketching — much faster than RCTD or Cell2location. Pure Python, no GPU needed.

filter_requirements:
  modalities: [gene]
  data_levels: [spot]

prerequisites:
  - Annotated single-cell reference dataset path (.h5ad file)
  - Cell type column name in the reference dataset (default celltype)

default_skill: true
---

# Fast Spot Deconvolution (FlashDeconv)

Ultra-fast reference-based cell type deconvolution using **FlashDeconv**. Estimates the proportion of each cell type in every spatial spot using random sketching — O(N) complexity, much faster than RCTD or Cell2location.

**Output**:
- `adata.obsm['deconv_weights']`: DataFrame (n_spots × n_celltypes) with proportions
- `adata.obs['celltype']`: Dominant cell type per spot

---

## Workflow

### Step 1: Load Data

```python
import scanpy as sc
import pandas as pd
import numpy as np

print("=" * 60)
print("STEP 1: Load Data")
print("=" * 60)

# Load spatial data
slice_id = 0  # <-- SET TARGET SLICE
slice_obj = session.get_slice(slice_id)
ad_spatial = slice_obj.adata.copy()

if not slice_obj.is_spot_level:
    raise ValueError("FlashDeconv requires spot-level data (Visium)")

# Load reference single-cell data
ad_sc = sc.read_h5ad('reference_scRNA.h5ad')  # <-- SET REFERENCE PATH

# IMPORTANT: Cell type column in reference
celltype_key = 'celltype'  # <-- SET CELL TYPE COLUMN

assert celltype_key in ad_sc.obs.columns, (
    f"Column '{celltype_key}' not found in reference. "
    f"Available: {list(ad_sc.obs.columns)}"
)

# Ensure unique names
ad_spatial.var_names_make_unique()
ad_sc.var_names_make_unique()

# Common genes
common_genes = list(set(ad_spatial.var_names) & set(ad_sc.var_names))
assert len(common_genes) >= 100, (
    f"Only {len(common_genes)} common genes. Need >= 100."
)

print(f"  Spatial: {ad_spatial.n_obs} spots, {ad_spatial.n_vars} genes")
print(f"  Reference: {ad_sc.n_obs} cells, {ad_sc.n_vars} genes")
print(f"  Cell types: {ad_sc.obs[celltype_key].nunique()}")
print(f"  Common genes: {len(common_genes)}")
```

### Step 2: Run FlashDeconv

```python
print("\n" + "=" * 60)
print("STEP 2: Run FlashDeconv")
print("=" * 60)

import flashdeconv

# Subset to common genes
ad_spatial_sub = ad_spatial[:, common_genes].copy()
ad_sc_sub = ad_sc[:, common_genes].copy()

# Ensure spatial coordinates in obsm
ad_spatial_sub.obsm['spatial'] = ad_spatial_sub.obs[['x', 'y']].to_numpy()

# Run deconvolution (results stored in ad_spatial_sub.obsm['flashdeconv'])
flashdeconv.tl.deconvolve(
    ad_spatial_sub,
    ad_sc_sub,
    cell_type_key=celltype_key,
    key_added='flashdeconv',
)

props = ad_spatial_sub.obsm['flashdeconv']
print(f"  Deconvolution complete")
print(f"  Result shape: {props.shape}")
```

### Step 3: Store Results

```python
print("\n" + "=" * 60)
print("STEP 3: Store Results")
print("=" * 60)

import pandas as pd

# Store proportions
props = ad_spatial_sub.obsm['flashdeconv']
if not isinstance(props, pd.DataFrame):
    # Convert to DataFrame with cell type names
    ct_names = ad_sc.obs[celltype_key].cat.categories.tolist() if hasattr(ad_sc.obs[celltype_key], 'cat') else sorted(ad_sc.obs[celltype_key].unique())
    props = pd.DataFrame(props, index=ad_spatial_sub.obs_names, columns=ct_names[:props.shape[1]])

slice_obj.adata.obsm['deconv_weights'] = props
slice_obj.adata.uns['has_deconv_weights'] = True

# Dominant cell type per spot
dominant = props.idxmax(axis=1)
slice_obj.adata.obs['celltype'] = dominant.values

print(f"  Stored deconv_weights in adata.obsm ({props.shape})")
print(f"  Stored dominant celltype in adata.obs['celltype']")

# Summary
print(f"\nCell type proportions summary:")
mean_props = props.mean(axis=0).sort_values(ascending=False)
for ct, prop in mean_props.items():
    print(f"  {ct}: {prop:.3f} (mean proportion)")
```

---

## Parameter Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `celltype_key` | `'celltype'` | Column name for cell types in reference |
| Reference path | — | Path to scRNA-seq .h5ad file |

## Notes

- FlashDeconv uses random sketching for O(N) time complexity — handles 1M+ spots in minutes.
- No GPU required. Pure Python.
- Same output format as RCTD — `adata.obsm['deconv_weights']` and `adata.obs['celltype']`.
- Reference dataset should have diverse cell types representing the tissue.
- Raw counts are preferred for both spatial and reference data.
