---
name: spatial-stats-neighborhood-enrichment
title: Neighborhood Enrichment Analysis
slug: spatial-stats-neighborhood-enrichment
description: Compute neighborhood enrichment z-scores to identify which cell types are spatially co-localized or depleted from each other's neighborhoods. Uses squidpy permutation testing on spatial neighbor graphs. Requires cell type annotations.

filter_requirements:
  num_slices: 1
  modalities: [gene/protein]
  data_levels: [cell/spot]

prerequisites:
  - Cell type annotations in the target slice (adata.obs['celltype'])
default_skill: true
---

# Neighborhood Enrichment Analysis

Compute **neighborhood enrichment z-scores** to identify which cell types are spatially enriched or depleted in each other's neighborhoods. Uses **squidpy**'s permutation-based test (`sq.gr.nhood_enrichment`) on a spatial neighbor graph.

**What it measures**:
- **Positive z-score**: Two cell types co-localize (found near each other more than expected by chance)
- **Negative z-score**: Two cell types avoid each other (found near each other less than expected)
- **Near zero**: No significant spatial relationship

**Output**:
- `adata.uns['celltype_nhood_enrichment']`: Z-score matrix and count matrix
- Heatmap visualization of cell type co-localization patterns

---

## Workflow

### Stage 1: Load and Validate

```python
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

print("=" * 60)
print("STAGE 1: Load and Validate")
print("=" * 60)

# IMPORTANT: Specify the target slice
slice_id = 0
slice_obj = session.get_slice(slice_id)
adata = slice_obj.adata.copy()

# Validate spatial coordinates
assert 'x' in adata.obs.columns and 'y' in adata.obs.columns, "Missing spatial coordinates"
adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()

# Validate cell type annotations
cluster_key = 'celltype'
assert cluster_key in adata.obs.columns, (
    f"Missing '{cluster_key}' column in adata.obs. "
    "Run cell type annotation first."
)
assert adata.obs[cluster_key].notna().all(), "Some cells have NaN cell type labels"

# Ensure categorical
adata.obs[cluster_key] = pd.Categorical(adata.obs[cluster_key])

n_types = adata.obs[cluster_key].nunique()
print(f"Data: {adata.n_obs} cells/spots, {adata.n_vars} genes")
print(f"Cell types ({n_types}): {', '.join(adata.obs[cluster_key].cat.categories[:10])}")
```

### Stage 2: Build Spatial Neighbor Graph

```python
print("\n" + "=" * 60)
print("STAGE 2: Build Spatial Neighbor Graph")
print("=" * 60)

# Build spatial neighbor graph using squidpy
# coord_type='generic' works for both cell-level and spot-level data
# n_neighs: number of nearest neighbors (6 for grid/Visium, 10-15 for irregular)
is_spot = slice_obj.is_spot_level
n_neighs = 6 if is_spot else 10

sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=n_neighs)

n_edges = adata.obsp['spatial_connectivities'].nnz
avg_neighbors = n_edges / adata.n_obs
print(f"Spatial neighbor graph built")
print(f"  n_neighs: {n_neighs}")
print(f"  Total edges: {n_edges:,}")
print(f"  Avg neighbors per cell: {avg_neighbors:.1f}")
```

### Stage 3: Compute Neighborhood Enrichment

```python
print("\n" + "=" * 60)
print("STAGE 3: Compute Neighborhood Enrichment")
print("=" * 60)

# Number of permutations for statistical testing
n_perms = 1000  # Default: 1000 (higher = more stable p-values but slower)
seed = 42

print(f"  Running permutation test ({n_perms} permutations)...")

sq.gr.nhood_enrichment(
    adata,
    cluster_key=cluster_key,
    n_perms=n_perms,
    seed=seed,
)

# Results stored in adata.uns['{cluster_key}_nhood_enrichment']
analysis_key = f"{cluster_key}_nhood_enrichment"
assert analysis_key in adata.uns, "Neighborhood enrichment did not produce results"

zscore_matrix = adata.uns[analysis_key]["zscore"]
count_matrix = adata.uns[analysis_key]["count"]

# Get cell type names (categories in the order used by squidpy)
ct_names = list(adata.obs[cluster_key].cat.categories)

print(f"Neighborhood enrichment computed")
print(f"  Z-score matrix: {zscore_matrix.shape}")
print(f"  Max enrichment: {np.nanmax(zscore_matrix):.2f}")
print(f"  Min enrichment (depletion): {np.nanmin(zscore_matrix):.2f}")
```

### Stage 4: Summarize Results

```python
print("\n" + "=" * 60)
print("STAGE 4: Summarize Results")
print("=" * 60)

# Create labeled DataFrame for easier interpretation
zscore_df = pd.DataFrame(
    zscore_matrix,
    index=ct_names,
    columns=ct_names,
)

# Find top enriched pairs (positive z-scores = co-localization)
pairs = []
for i in range(len(ct_names)):
    for j in range(i + 1, len(ct_names)):
        z = zscore_matrix[i, j]
        if not np.isnan(z):
            pairs.append((ct_names[i], ct_names[j], z))

pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print("Top enriched pairs (co-localized):")
enriched = [p for p in pairs if p[2] > 0]
for ct1, ct2, z in enriched[:10]:
    print(f"  {ct1} <-> {ct2}: z={z:.2f}")

print("\nTop depleted pairs (spatially separated):")
depleted = [p for p in pairs if p[2] < 0]
for ct1, ct2, z in depleted[:10]:
    print(f"  {ct1} <-> {ct2}: z={z:.2f}")

# Self-enrichment (diagonal) — how much each type clusters with itself
print("\nSelf-enrichment (spatial clustering of same type):")
for i, ct in enumerate(ct_names):
    z = zscore_matrix[i, i]
    if not np.isnan(z):
        label = "clustered" if z > 0 else "dispersed"
        print(f"  {ct}: z={z:.2f} ({label})")
```

### Stage 5: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 5: Store Results")
print("=" * 60)

# Transfer results back to session slice
slice_obj.adata.uns[analysis_key] = adata.uns[analysis_key]

# Also store the labeled z-score DataFrame for convenience
slice_obj.adata.uns['nhood_enrichment_zscore'] = zscore_df

print(f"Added '{analysis_key}' to slice_obj.adata.uns")
print(f"  Contains 'zscore' matrix ({zscore_matrix.shape}) and 'count' matrix")
print(f"Added 'nhood_enrichment_zscore' (labeled DataFrame) to slice_obj.adata.uns")
print(f"\nNeighborhood enrichment analysis complete!")
```

## Visualization

### Z-Score Heatmap

```python
import matplotlib.pyplot as plt

zscore_df = slice_obj.adata.uns['nhood_enrichment_zscore']

fig, ax = plt.subplots(figsize=(10, 8))
vmax = max(abs(np.nanmin(zscore_df.values)), abs(np.nanmax(zscore_df.values)))

im = ax.imshow(zscore_df.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xticks(range(len(zscore_df.columns)))
ax.set_yticks(range(len(zscore_df.index)))
ax.set_xticklabels(zscore_df.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(zscore_df.index, fontsize=8)
ax.set_title('Neighborhood Enrichment (z-scores)')

plt.colorbar(im, ax=ax, label='z-score', shrink=0.8)
plt.tight_layout()
plt.show()
```



