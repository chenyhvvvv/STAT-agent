---
name: integration-bbknn
title: Batch Integration (BBKNN)
slug: integration-bbknn
description: Correct batch effects across multiple slices using BBKNN (Batch Balanced K-Nearest Neighbors). Modifies the neighbor graph to connect cells across batches, enabling joint clustering and visualization. Lightweight and fast.

filter_requirements:
  modalities: [gene]

prerequisites:
  - Multiple slices loaded in session

default_skill: false
---

# Batch Integration (BBKNN)

Correct **batch effects** across multiple slices using **BBKNN** (Batch Balanced K-Nearest Neighbors). Instead of correcting the expression matrix, BBKNN modifies the **neighbor graph** to balance connections across batches — enabling joint clustering and UMAP visualization.

**Output**: Batch-corrected neighbor graph in `adata.obsp`, joint UMAP in `adata.obsm['X_umap']`.

---

## Workflow

### Stage 1: Collect and Concatenate Slices

```python
import numpy as np
import pandas as pd
import scanpy as sc

print("=" * 60)
print("STAGE 1: Collect and Concatenate Slices")
print("=" * 60)

# Collect all slices
slice_ids = session.get_slice_ids()
assert len(slice_ids) >= 2, f"Need >= 2 slices for integration, got {len(slice_ids)}"

adatas = []
for sid in slice_ids:
    s = session.get_slice(sid)
    ad = s.adata.copy()
    ad.obs['batch'] = f'slice_{sid}'
    # Slice-id marker for safe positional merge-back (avoids obs_names collisions)
    ad.obs['_slice_id'] = sid
    adatas.append(ad)
    print(f"  Slice {sid}: {ad.n_obs} cells, {ad.n_vars} genes")

# Concatenate
adata = sc.concat(adatas, join='inner', label='batch', keys=[f'slice_{sid}' for sid in slice_ids])
adata.obs_names_make_unique()
print(f"\n  Combined: {adata.n_obs} cells, {adata.n_vars} genes, {len(slice_ids)} batches")
```

### Stage 2: Preprocessing

```python
print("\n" + "=" * 60)
print("STAGE 2: Preprocessing")
print("=" * 60)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='batch')
adata = adata[:, adata.var['highly_variable']].copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50)

print(f"  HVGs: {sum(adata.var['highly_variable'] if 'highly_variable' in adata.var else [])}")
print(f"  PCA computed (50 components)")
```

### Stage 3: Run BBKNN

```python
print("\n" + "=" * 60)
print("STAGE 3: Run BBKNN")
print("=" * 60)

import bbknn

bbknn.bbknn(adata, batch_key='batch', n_pcs=30)

# Compute joint UMAP and clustering
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=1.0)

print(f"  BBKNN integration complete")
print(f"  Leiden clusters: {adata.obs['leiden'].nunique()}")
```

### Stage 4: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 4: Store Results Back to Slices")
print("=" * 60)

# Store integrated results back to each slice — use the _slice_id marker
# we set in Stage 1, NOT obs_names (which can collide across slices and
# silently corrupt the assignment).
for sid in slice_ids:
    s = session.get_slice(sid)
    slice_mask = (adata.obs['_slice_id'] == sid).values
    n_in_slice = int(slice_mask.sum())
    if n_in_slice == s.adata.n_obs:
        # Positional copy in original order
        s.adata.obs['leiden_integrated'] = adata.obs.loc[slice_mask, 'leiden'].values
        print(f"  Slice {sid}: stored integrated leiden clusters ({n_in_slice} cells)")
    else:
        print(f"  Slice {sid}: WARNING combined has {n_in_slice} cells but slice has {s.adata.n_obs}; skipping")

# Also store full integrated adata in uns of first slice for reference
session.get_slice(slice_ids[0]).adata.uns['integration_bbknn'] = {
    'method': 'bbknn',
    'n_slices': len(slice_ids),
    'n_cells_total': adata.n_obs,
    'n_clusters': adata.obs['leiden'].nunique(),
}

print(f"\nBBKNN integration complete. Joint clusters in adata.obs['leiden_integrated']")
```

## Visualization

### UMAP by Batch

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
| `batch_key` | `'batch'` | Any obs column | Batch identifier |
| `n_pcs` | `30` | 10-50 | PCs for BBKNN |
| `resolution` | `1.0` | 0.1-2.0 | Leiden clustering resolution |

## Notes

- BBKNN modifies the **neighbor graph**, not the expression matrix — this preserves biological signal.
- Lightweight and fast compared to Harmony or scVI.
- Joint Leiden clusters are stored as `leiden_integrated` to avoid overwriting existing cluster annotations.
- For expression-level correction (e.g., for DE analysis), consider Harmony or Scanorama instead.
