---
name: annotation-tangram
title: Cell Type Annotation via Spatial Mapping (Tangram)
slug: annotation-tangram
description: Map single-cell reference annotations onto spatial transcriptomics data using Tangram deep learning alignment. Projects cell type labels from scRNA-seq reference to spatial spots/cells. Works for both cell-level and spot-level (deconvolution-like) data.

filter_requirements:
  modalities: [gene]
  data_levels: [cell/spot]

prerequisites:
  - Annotated single-cell reference dataset path (.h5ad file)
  - Cell type column name in reference dataset (default celltype)

default_skill: false
---

# Cell Type Annotation via Spatial Mapping (Tangram)

Map **single-cell reference annotations** onto spatial data using **Tangram** — a deep learning method that aligns scRNA-seq cells to spatial locations. Unlike correlation-based methods, Tangram learns a probabilistic mapping that respects gene expression patterns.

**Output**:
- Cell-level: `adata.obs['celltype']` — predicted cell type
- Spot-level: `adata.obsm['tangram_ct_pred']` — cell type proportions per spot + `adata.obs['celltype']` (dominant)

---

## Workflow

### Stage 1: Load Data

```python
import scanpy as sc
import pandas as pd
import numpy as np

print("=" * 60)
print("STAGE 1: Load Data")
print("=" * 60)

# Load spatial data
slice_id = 0  # <-- SET TARGET SLICE
slice_obj = session.get_slice(slice_id)
adata_sp = slice_obj.adata.copy()

# Load reference
adata_sc = sc.read_h5ad('reference_scRNA.h5ad')  # <-- SET REFERENCE PATH

# IMPORTANT: Cell type column
celltype_key = 'celltype'  # <-- SET CELL TYPE COLUMN

assert celltype_key in adata_sc.obs.columns, (
    f"Column '{celltype_key}' not found in reference."
)

# Ensure unique names
adata_sp.var_names_make_unique()
adata_sc.var_names_make_unique()

print(f"  Spatial: {adata_sp.n_obs} cells/spots, {adata_sp.n_vars} genes")
print(f"  Reference: {adata_sc.n_obs} cells, {adata_sc.n_vars} genes")
print(f"  Cell types: {adata_sc.obs[celltype_key].nunique()}")
```

### Stage 2: Prepare and Find Marker Genes

```python
print("\n" + "=" * 60)
print("STAGE 2: Prepare Marker Genes")
print("=" * 60)

import tangram as tg

# Normalize reference if needed
adata_sc_pp = adata_sc.copy()
sc.pp.normalize_total(adata_sc_pp, target_sum=1e4)
sc.pp.log1p(adata_sc_pp)

# Find markers for mapping
sc.tl.rank_genes_groups(adata_sc_pp, groupby=celltype_key, method='wilcoxon')
markers_df = pd.DataFrame(adata_sc_pp.uns['rank_genes_groups']['names']).iloc[:100, :]
markers = list(set(markers_df.values.flatten()))

# Filter to common genes
common = set(adata_sp.var_names) & set(adata_sc.var_names) & set(markers)
markers = list(common)
print(f"  Marker genes for mapping: {len(markers)}")

# Prepare adatas
tg.pp_adatas(adata_sc, adata_sp, genes=markers)
print(f"  Adatas prepared for Tangram")
```

### Stage 3: Run Tangram Mapping

```python
print("\n" + "=" * 60)
print("STAGE 3: Run Tangram Mapping")
print("=" * 60)

# Map cells to space
ad_map = tg.map_cells_to_space(
    adata_sc,
    adata_sp,
    mode='cells',
    density_prior='rna_count_based',
    num_epochs=500,
)

print(f"  Mapping complete")
print(f"  Mapping matrix shape: {ad_map.X.shape}")

# Project annotations
tg.project_cell_annotations(ad_map, adata_sp, annotation=celltype_key)

ct_cols = [c for c in adata_sp.obsm['tangram_ct_pred'].columns]
print(f"  Projected {len(ct_cols)} cell types")
```

### Stage 4: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 4: Store Results")
print("=" * 60)

# Get predictions
ct_pred = adata_sp.obsm['tangram_ct_pred']

# Dominant cell type
dominant_ct = ct_pred.idxmax(axis=1)
slice_obj.adata.obs['celltype'] = dominant_ct.values

# For spot-level data, also store proportions
if slice_obj.is_spot_level:
    slice_obj.adata.obsm['tangram_ct_pred'] = ct_pred
    slice_obj.adata.obsm['deconv_weights'] = ct_pred
    slice_obj.adata.uns['has_deconv_weights'] = True
    print(f"  Stored cell type proportions in adata.obsm['deconv_weights']")

slice_obj.adata.uns['annotation_params'] = {
    'method': 'tangram',
    'n_markers': len(markers),
    'n_celltypes': len(ct_cols),
    'celltypes': ct_cols,
}

print(f"  Stored celltype in adata.obs['celltype']")

# Summary
print(f"\nCell type distribution:")
for ct in sorted(dominant_ct.unique()):
    n = (dominant_ct == ct).sum()
    pct = n / len(dominant_ct) * 100
    print(f"  {ct}: {n} ({pct:.1f}%)")
```

---

## Parameter Guide

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `celltype_key` | `'celltype'` | Any obs column in reference | Cell type column |
| `mode` | `'cells'` | `'cells'`, `'clusters'` | Mapping mode |
| `num_epochs` | 500 | 200-1000 | Training epochs |
| `density_prior` | `'rna_count_based'` | `'rna_count_based'`, `'uniform'` | Density prior |

## Notes

- Tangram works for both **cell-level** (annotation) and **spot-level** (deconvolution-like) data.
- For spot data, proportions are stored in `adata.obsm['deconv_weights']` (same format as RCTD/FlashDeconv).
- GPU is used automatically if available (PyTorch-based).
- Reference dataset should represent the cell types present in the spatial tissue.
