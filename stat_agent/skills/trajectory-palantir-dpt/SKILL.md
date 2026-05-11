---
name: trajectory-pseudotime
title: Pseudotime Trajectory Analysis (Palantir / DPT)
slug: trajectory-pseudotime
description: Infer cell developmental trajectories and pseudotime ordering using expression-based methods. Palantir uses diffusion maps and random walks for branching trajectories. DPT (Diffusion Pseudotime) is a lightweight alternative using scanpy. No RNA velocity data required.

filter_requirements:
  num_slices: 1
  modalities: [gene]
  data_levels: [cell]

prerequisites:
  - "Root cell type or root cell hint — REQUIRED. Specify the least differentiated population (e.g., stem cells, progenitors, basal cells). Without a biologically motivated root, pseudotime is meaningless: the auto-fallback (extreme of DC1) gives a numerically valid but biologically arbitrary ordering."
  - Cell type annotations (recommended for interpreting trajectory)

default_skill: false
---

# Pseudotime Trajectory Analysis

Infer **cell developmental trajectories** and **pseudotime ordering** from gene expression. Two modes:

- **Palantir**: Diffusion maps + multiscale random walks. Detects branching trajectories and fate probabilities. Best for complex developmental processes.
- **DPT**: Diffusion Pseudotime via scanpy. Lightweight, fast. Good for simple linear trajectories.

**No RNA velocity data required** — works purely from expression.

**Output**: `adata.obs['pseudotime']` for spatial visualization of developmental gradients.

---

## Workflow

### Stage 1: Load and Preprocess

```python
import numpy as np
import pandas as pd
import scanpy as sc

print("=" * 60)
print("STAGE 1: Load and Preprocess")
print("=" * 60)

# IMPORTANT: Target slice
slice_id = 0  # <-- SET TARGET SLICE
slice_obj = session.get_slice(slice_id)
adata = slice_obj.adata.copy()

print(f"  Data: {adata.n_obs} cells, {adata.n_vars} genes")

# Preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']].copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)

print(f"  Preprocessed: {adata.n_obs} cells, {sum(adata.var['highly_variable'] if 'highly_variable' in adata.var else [])} HVGs")
print(f"  PCA and neighbors computed")
```

### Mode A: Palantir Trajectory

```python
print("\n" + "=" * 60)
print("MODE A: Palantir Trajectory")
print("=" * 60)

import palantir

# Diffusion maps
palantir.utils.run_diffusion_maps(adata, n_components=10)
palantir.utils.determine_multiscale_space(adata)
print(f"  Diffusion maps and multiscale space computed")

# IMPORTANT: Root cell selection
# Option 1: Specify root cell type → pick cell at extreme of first diffusion component
root_celltype = 'Stem'  # <-- SET ROOT CELL TYPE (or None for auto)

if root_celltype and 'celltype' in slice_obj.adata.obs.columns:
    ct_mask = slice_obj.adata.obs['celltype'] == root_celltype
    if ct_mask.any():
        # Pick cell in root_celltype with most extreme DC1 value
        dc1 = adata.obsm['DM_EigenVectors'][:, 0]
        dc1_masked = np.where(ct_mask.values, dc1, np.nan)
        root_idx = np.nanargmin(dc1_masked)
        start_cell = adata.obs_names[root_idx]
        print(f"  Root cell: {start_cell} (from {root_celltype})")
    else:
        print(f"  Warning: '{root_celltype}' not found, using auto-selection")
        root_idx = np.argmin(adata.obsm['DM_EigenVectors'][:, 0])
        start_cell = adata.obs_names[root_idx]
else:
    # Auto-select: extreme of first diffusion component
    root_idx = np.argmin(adata.obsm['DM_EigenVectors'][:, 0])
    start_cell = adata.obs_names[root_idx]
    print(f"  Root cell (auto): {start_cell}")

# Run Palantir
pr_res = palantir.core.run_palantir(
    adata,
    early_cell=start_cell,
    num_waypoints=500,
)

print(f"  Pseudotime range: [{adata.obs['palantir_pseudotime'].min():.3f}, {adata.obs['palantir_pseudotime'].max():.3f}]")
print(f"  Entropy range: [{adata.obs['palantir_entropy'].min():.3f}, {adata.obs['palantir_entropy'].max():.3f}]")

# Store results
slice_obj.adata.obs['pseudotime'] = adata.obs['palantir_pseudotime'].values
slice_obj.adata.obs['palantir_entropy'] = adata.obs['palantir_entropy'].values
if 'palantir_fate_probabilities' in adata.obsm:
    slice_obj.adata.obsm['palantir_fate_probs'] = adata.obsm['palantir_fate_probabilities']

slice_obj.adata.uns['trajectory_params'] = {
    'method': 'palantir',
    'root_cell': start_cell,
    'root_celltype': root_celltype,
    'n_waypoints': 500,
}

print(f"  Stored pseudotime in adata.obs['pseudotime']")
```

### Mode B: DPT (Lightweight Alternative)

```python
print("\n" + "=" * 60)
print("MODE B: Diffusion Pseudotime (DPT)")
print("=" * 60)

# Compute diffusion map
sc.tl.diffmap(adata, n_comps=15)

# IMPORTANT: Root cell selection (same logic as above)
root_celltype = 'Stem'  # <-- SET ROOT CELL TYPE (or None for auto)

if root_celltype and 'celltype' in slice_obj.adata.obs.columns:
    ct_mask = slice_obj.adata.obs['celltype'] == root_celltype
    if ct_mask.any():
        dc1 = adata.obsm['DM_EigenVectors'][:, 0]
        dc1_masked = np.where(ct_mask.values, dc1, np.nan)
        root_idx = int(np.nanargmin(dc1_masked))
    else:
        root_idx = int(np.argmin(adata.obsm['DM_EigenVectors'][:, 0]))
else:
    root_idx = int(np.argmin(adata.obsm['DM_EigenVectors'][:, 0]))

adata.uns['iroot'] = root_idx
sc.tl.dpt(adata, n_dcs=10)

print(f"  Root cell index: {root_idx}")
print(f"  Pseudotime range: [{adata.obs['dpt_pseudotime'].min():.3f}, {adata.obs['dpt_pseudotime'].max():.3f}]")

# Store results
slice_obj.adata.obs['pseudotime'] = adata.obs['dpt_pseudotime'].values
slice_obj.adata.uns['trajectory_params'] = {
    'method': 'dpt',
    'root_index': root_idx,
    'root_celltype': root_celltype,
}

print(f"  Stored pseudotime in adata.obs['pseudotime']")
```

## Visualization

### Spatial Pseudotime Map

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
x = slice_obj.adata.obs['x'].values
y = slice_obj.adata.obs['y'].values
pt = slice_obj.adata.obs['pseudotime'].values

# Handle NaN/inf
valid = np.isfinite(pt)
scatter = ax.scatter(x[valid], y[valid], c=pt[valid], cmap='viridis', s=3, alpha=0.8)
ax.set_title('Pseudotime (spatial)')
ax.set_aspect('equal')
ax.invert_yaxis()
plt.colorbar(scatter, ax=ax, label='Pseudotime', shrink=0.7)
plt.tight_layout()
plt.show()
```

---

## Parameter Guide

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `root_celltype` | None | Any celltype name | Cell type to use as trajectory root |
| Method | Palantir | `palantir`, `dpt` | Trajectory inference method |
| `n_top_genes` | 2000 | 1000-5000 | HVGs for preprocessing |
| `num_waypoints` | 500 | 200-1000 | Palantir waypoints (more = slower but finer) |

## Notes

- **Palantir** is recommended for branching trajectories (multiple terminal states). It also provides entropy (differentiation potential) and fate probabilities.
- **DPT** is a lightweight fallback — fast, simple, built into scanpy. Best for linear or simple trajectories.
- Root cell selection is critical. Provide the **least differentiated** cell type (e.g., stem cells, progenitors).
- Results in `adata.obs['pseudotime']` can be visualized spatially to see developmental gradients across tissue.
