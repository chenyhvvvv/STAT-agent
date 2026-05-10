---
name: integration-scanorama
title: Batch Integration (Scanorama)
slug: integration-scanorama
description: Correct batch effects across multiple slices using Scanorama panoramic stitching. Finds mutual nearest neighbors across datasets to learn a shared embedding, preserving biological variation while removing technical differences.

filter_requirements:
  modalities: [gene]

prerequisites:
  - Multiple slices loaded in session

default_skill: false
---

# Batch Integration (Scanorama)

Correct **batch effects** across multiple slices using **Scanorama** — a panoramic stitching approach that finds mutual nearest neighbors across datasets. Preserves biological variation better than simple methods, and works well even when batches share few cell types.

**Output**: Corrected embedding in `adata.obsm['X_scanorama']`, joint UMAP and clustering.

---

## Workflow

### Stage 1: Collect and Prepare Slices

```python
import numpy as np
import pandas as pd
import scanpy as sc

print("=" * 60)
print("STAGE 1: Collect and Prepare Slices")
print("=" * 60)

slice_ids = session.get_slice_ids()
assert len(slice_ids) >= 2, f"Need >= 2 slices, got {len(slice_ids)}"

adatas = []
for sid in slice_ids:
    s = session.get_slice(sid)
    ad = s.adata.copy()
    ad.obs['batch'] = f'slice_{sid}'
    adatas.append(ad)
    print(f"  Slice {sid}: {ad.n_obs} cells, {ad.n_vars} genes")

# Find common genes
common_genes = set(adatas[0].var_names)
for ad in adatas[1:]:
    common_genes &= set(ad.var_names)
common_genes = sorted(common_genes)
print(f"\n  Common genes: {len(common_genes)}")

# Subset and normalize each
adatas_pp = []
for ad in adatas:
    ad_sub = ad[:, common_genes].copy()
    sc.pp.normalize_total(ad_sub, target_sum=1e4)
    sc.pp.log1p(ad_sub)
    adatas_pp.append(ad_sub)
```

### Stage 2: Run Scanorama

```python
print("\n" + "=" * 60)
print("STAGE 2: Run Scanorama Integration")
print("=" * 60)

import numpy as np
import scanorama

# Integrate. NOTE: correct_scanpy returns NEW adata objects with
# obsm['X_scanorama']; the input list is NOT modified in place
# (despite the docstring suggesting otherwise).
corrected = scanorama.correct_scanpy(adatas_pp, return_dimred=True)

# Concatenate the *corrected* adatas. sc.concat drops obsm by default,
# so we re-attach X_scanorama explicitly afterwards.
X_scanorama = np.concatenate([a.obsm['X_scanorama'] for a in corrected], axis=0)
adata = sc.concat(corrected, join='inner', label='batch',
                  keys=[f'slice_{sid}' for sid in slice_ids])
adata.obsm['X_scanorama'] = X_scanorama
adata.obs_names_make_unique()

print(f"  Scanorama integration complete")
print(f"  Combined: {adata.n_obs} cells")

# Compute joint visualization
sc.pp.neighbors(adata, use_rep='X_scanorama', n_neighbors=15)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=1.0)

print(f"  Joint Leiden clusters: {adata.obs['leiden'].nunique()}")
```

### Stage 3: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 3: Store Results Back to Slices")
print("=" * 60)

for sid in slice_ids:
    s = session.get_slice(sid)
    batch_mask = adata.obs['batch'] == f'slice_{sid}'
    batch_cells = adata.obs_names[batch_mask]

    common_cells = s.adata.obs_names.isin(batch_cells)
    if common_cells.any():
        batch_adata = adata[batch_mask]
        s.adata.obs['leiden_integrated'] = pd.Series(
            batch_adata.obs['leiden'].values,
            index=batch_adata.obs_names
        ).reindex(s.adata.obs_names)
        print(f"  Slice {sid}: stored integrated clusters")

session.get_slice(slice_ids[0]).adata.uns['integration_scanorama'] = {
    'method': 'scanorama',
    'n_slices': len(slice_ids),
    'n_cells_total': adata.n_obs,
    'n_common_genes': len(common_genes),
    'n_clusters': adata.obs['leiden'].nunique(),
}

print(f"\nScanorama integration complete.")
```

## Visualization

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc.pl.umap(adata, color='batch', ax=axes[0], show=False, title='By Batch')
sc.pl.umap(adata, color='leiden', ax=axes[1], show=False, title='Joint Clusters')
plt.tight_layout()
plt.show()
```

---

## Parameter Guide

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `resolution` | `1.0` | 0.1-2.0 | Leiden clustering resolution |
| `n_neighbors` | `15` | 5-30 | Neighbors for UMAP |

## Notes

- Scanorama corrects both the expression matrix and produces a shared embedding.
- Works well even when batches share few cell types (unlike Harmony).
- For graph-only correction (lighter), consider BBKNN instead.
