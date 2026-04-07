---
name: spatial-domain-detection
title: Spatial Domain Detection (SpaGCN)
slug: spatial-domain-detection
description: Identify spatial domains in spot-level spatial transcriptomics data using SpaGCN, integrating gene expression, spatial location, and H&E histology image features.

filter_requirements:
  num_slices: 1
  modalities: [gene]
  data_levels: [spot]

prerequisites:
  - Number of expected spatial domains (e.g. 7)
  - H&E image (in the session) is recommended but optional of the target slice
default_skill: true
---

# Spatial Domain Detection Using SpaGCN

Identify spatial domains in spot-level spatial transcriptomics data (e.g. Visium) using **SpaGCN** (Spatial Graph Convolutional Network). SpaGCN integrates gene expression, spatial coordinates, and optionally H&E histology image features to detect tissue domains.

## Overview
**Output**:
- `adata.obs['spatial_domain']`: Domain label per spot

## Workflow

### Stage 1: Load and Validate

```python
import numpy as np
import scanpy as sc
import SpaGCN as spg
import random
import torch

# IMPORTANT: Specify the target slice (e.g. slice 0)
slice_obj = session.get_slice(0)
adata = slice_obj.adata.copy()

print("=" * 60)
print("STAGE 1: Load and Validate")
print("=" * 60)

# Ensure unique names
adata.var_names_make_unique()
adata.obs_names_make_unique()

# Validate spatial coordinates
assert 'x' in adata.obs.columns and 'y' in adata.obs.columns, "Missing spatial coordinates"

# Fill NaN coordinates if any
adata.obs['x'] = adata.obs['x'].fillna(adata.obs['x'].mean())
adata.obs['y'] = adata.obs['y'].fillna(adata.obs['y'].mean())

x_pixel = adata.obs['x'].astype(int).tolist()
y_pixel = adata.obs['y'].astype(int).tolist()

# Check for H&E image
img = None
if slice_obj.images:
    img_key = list(slice_obj.images.keys())[0]
    img = slice_obj.images[img_key]
    # SpaGCN expects (height, width, channels)
    if img.ndim == 3 and img.shape[0] <= 4:
        img = img.transpose(1, 2, 0)
    print(f"✓ H&E image loaded: {img.shape}")
else:
    print("⚠ No H&E image found, using spatial coordinates only")

print(f"✓ Data: {adata.n_obs} spots, {adata.n_vars} genes")
```

### Stage 2: Build Adjacency Matrix

```python
print("\n" + "=" * 60)
print("STAGE 2: Build Adjacency Matrix")
print("=" * 60)

if img is not None:
    # Integrate histology features (recommended)
    adj = spg.calculate_adj_matrix(
        x=x_pixel, y=y_pixel,
        x_pixel=x_pixel, y_pixel=y_pixel,
        image=img, beta=49, alpha=1, histology=True
    )
    print("✓ Adjacency matrix built with histology integration")
else:
    # Spatial coordinates only
    adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    print("✓ Adjacency matrix built with spatial coordinates only")
```

### Stage 3: Preprocess and Find Parameters

```python
print("\n" + "=" * 60)
print("STAGE 3: Preprocess and Find Parameters")
print("=" * 60)

# Prefilter and normalize
spg.prefilter_genes(adata, min_cells=3)
spg.prefilter_specialgenes(adata)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Densify if sparse
if hasattr(adata.X, 'toarray'):
    adata.X = adata.X.toarray()

# Find optimal l parameter
p = 0.5
l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
print(f"✓ Optimal l = {l:.4f}")

# Search resolution for target number of clusters
n_clusters = 7  # IMPORTANT: Adjust based on expected number of domains
r_seed = t_seed = n_seed = 100

res = spg.search_res(
    adata, adj, l, n_clusters,
    start=0.7, step=0.1, tol=5e-3,
    lr=0.05, max_epochs=20,
    r_seed=r_seed, t_seed=t_seed, n_seed=n_seed
)
print(f"✓ Optimal resolution = {res:.4f}")
```

### Stage 4: Run SpaGCN and Refine

```python
print("\n" + "=" * 60)
print("STAGE 4: Run SpaGCN")
print("=" * 60)

# Train model
clf = spg.SpaGCN()
clf.set_l(l)
random.seed(r_seed)
torch.manual_seed(t_seed)
np.random.seed(n_seed)

clf.train(
    adata, adj, init_spa=True, init="louvain",
    res=res, tol=5e-3, lr=0.05, max_epochs=200
)

y_pred, prob = clf.predict()
adata.obs["spatial_domain_raw"] = y_pred
adata.obs["spatial_domain_raw"] = adata.obs["spatial_domain_raw"].astype('category')
print(f"✓ Raw prediction: {len(set(y_pred))} domains")

# Refine predictions (shape="hexagon" for Visium, "square" for ST)
adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
refined_pred = spg.refine(
    sample_id=adata.obs.index.tolist(),
    pred=adata.obs["spatial_domain_raw"].tolist(),
    dis=adj_2d, shape="hexagon"
)
adata.obs["spatial_domain"] = refined_pred
adata.obs["spatial_domain"] = adata.obs["spatial_domain"].astype('category')
print(f"✓ Refined prediction: {adata.obs['spatial_domain'].nunique()} domains")
```

### Stage 5: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 5: Storing Results")
print("=" * 60)

# Transfer domain labels back to session slice
slice_obj.adata.obs['spatial_domain'] = adata.obs['spatial_domain'].values

print(f"✓ Added 'spatial_domain' to slice_obj.adata.obs")
print(f"  {slice_obj.adata.obs['spatial_domain'].nunique()} domains detected")
print(f"\nDomain distribution:")
print(slice_obj.adata.obs['spatial_domain'].value_counts().sort_index())
```
