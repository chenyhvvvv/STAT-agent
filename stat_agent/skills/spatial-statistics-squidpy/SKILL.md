---
name: spatial-statistics
title: Spatial Statistics Analysis
slug: spatial-statistics
description: Compute spatial statistics including Moran's I (spatial autocorrelation of genes), Ripley's K (spatial point pattern of cell types), co-occurrence analysis, and centrality scores. Uses squidpy for permutation-based spatial statistical testing on neighbor graphs.

filter_requirements:
  modalities: [gene/protein]
  data_levels: [cell/spot]

prerequisites:
  - "For gene-level analysis (Moran's I): gene names of interest"
  - "For cell-type-level analysis (Ripley's K, co-occurrence, centrality): cell type annotations in adata.obs['celltype']"

default_skill: true
---

# Spatial Statistics Analysis

Compute **spatial statistics** to quantify spatial patterns in gene expression and cell type distributions. Includes multiple analysis modes:

- **Moran's I**: Spatial autocorrelation — which genes are spatially structured?
- **Ripley's K/L**: Spatial point pattern — are cell types clustered, dispersed, or random?
- **Co-occurrence**: At what distances do cell types co-occur?
- **Centrality scores**: Network centrality of cell types in the spatial graph

**Output**: Results stored in `adata.uns['spatial_stats_*']` and `adata.obs` for visualization.

---

## Workflow

### Stage 1: Load and Build Spatial Graph

```python
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

print("=" * 60)
print("STAGE 1: Load Data and Build Spatial Graph")
print("=" * 60)

# IMPORTANT: Target slice
slice_id = 0  # <-- SET TARGET SLICE
slice_obj = session.get_slice(slice_id)
adata = slice_obj.adata.copy()

# Ensure spatial coordinates in obsm
adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()

# Build spatial neighbor graph
is_spot = slice_obj.is_spot_level
n_neighs = 6 if is_spot else 10
sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=n_neighs)

print(f"  Data: {adata.n_obs} cells/spots, {adata.n_vars} genes")
print(f"  Spatial graph: {n_neighs} neighbors per cell")
```

### Mode A: Moran's I — Spatial Autocorrelation of Genes

Identify genes with **spatially structured expression** (not randomly distributed).

```python
print("\n" + "=" * 60)
print("MODE A: Moran's I — Spatial Autocorrelation")
print("=" * 60)

# Normalize for testing
adata_norm = adata.copy()
sc.pp.normalize_total(adata_norm, target_sum=1e4)
sc.pp.log1p(adata_norm)

# OPTION 1: Test specific genes
# genes_to_test = ['GENE1', 'GENE2']  # <-- SET GENES
# adata_test = adata_norm[:, genes_to_test]

# OPTION 2: Test all genes (or HVGs for speed)
sc.pp.highly_variable_genes(adata_norm, n_top_genes=2000)
adata_test = adata_norm[:, adata_norm.var['highly_variable']]

# Transfer spatial graph
adata_test.obsp['spatial_connectivities'] = adata_norm.obsp['spatial_connectivities']
adata_test.obsp['spatial_distances'] = adata_norm.obsp['spatial_distances']

sq.gr.spatial_autocorr(adata_test, mode='moran', n_jobs=1)

# Results in adata_test.uns['moranI']
moranI = adata_test.uns['moranI'].sort_values('I', ascending=False)

print(f"  Tested {len(moranI)} genes")
sig_genes = moranI[moranI['pval_norm_fdr_bh'] < 0.05]
print(f"  Spatially variable (FDR<0.05): {len(sig_genes)} genes")
print(f"\nTop 20 spatially autocorrelated genes:")
for i, (gene, row) in enumerate(moranI.head(20).iterrows()):
    print(f"  {i+1}. {gene}: I={row['I']:.4f}, FDR={row['pval_norm_fdr_bh']:.2e}")

# Store results
slice_obj.adata.uns['moranI'] = moranI
print(f"\nStored in adata.uns['moranI']")
```

### Mode B: Ripley's K/L — Cell Type Spatial Distribution

Test whether cell types are **clustered**, **dispersed**, or **randomly distributed**.

```python
print("\n" + "=" * 60)
print("MODE B: Ripley's K/L Function")
print("=" * 60)

cluster_key = 'celltype'  # <-- SET CLUSTER KEY
assert cluster_key in adata.obs.columns, f"Missing '{cluster_key}' column"
adata.obs[cluster_key] = pd.Categorical(adata.obs[cluster_key])

sq.gr.ripley(adata, cluster_key=cluster_key, mode='L')

# Results in adata.uns['{cluster_key}_ripley_L']
ripley_key = f'{cluster_key}_ripley_L'
ripley_results = adata.uns[ripley_key]
print(f"  Computed Ripley's L for {adata.obs[cluster_key].nunique()} cell types")

# Store results
slice_obj.adata.uns[ripley_key] = ripley_results
print(f"  Stored in adata.uns['{ripley_key}']")
```

### Mode C: Co-occurrence Analysis

Measure **pairwise co-occurrence probabilities** between cell types at varying distances.

```python
print("\n" + "=" * 60)
print("MODE C: Co-occurrence Analysis")
print("=" * 60)

cluster_key = 'celltype'  # <-- SET CLUSTER KEY
assert cluster_key in adata.obs.columns, f"Missing '{cluster_key}' column"
adata.obs[cluster_key] = pd.Categorical(adata.obs[cluster_key])

sq.gr.co_occurrence(adata, cluster_key=cluster_key)

# Results in adata.uns['{cluster_key}_co_occurrence']
co_key = f'{cluster_key}_co_occurrence'
co_results = adata.uns[co_key]
ct_names = adata.obs[cluster_key].cat.categories.tolist()
print(f"  Computed co-occurrence for {len(ct_names)} cell types")

# Store results
slice_obj.adata.uns[co_key] = co_results
print(f"  Stored in adata.uns['{co_key}']")
```

### Mode D: Centrality Scores

Compute **network centrality** (degree, closeness, betweenness) for each cell type.

```python
print("\n" + "=" * 60)
print("MODE D: Centrality Scores")
print("=" * 60)

cluster_key = 'celltype'  # <-- SET CLUSTER KEY
assert cluster_key in adata.obs.columns, f"Missing '{cluster_key}' column"
adata.obs[cluster_key] = pd.Categorical(adata.obs[cluster_key])

sq.gr.centrality_scores(adata, cluster_key=cluster_key)

# Results in adata.uns['{cluster_key}_centrality_scores']
cent_key = f'{cluster_key}_centrality_scores'
centrality_df = adata.uns[cent_key]
print(f"  Centrality scores computed for {len(centrality_df)} cell types")
print(centrality_df.to_string())

# Store results
slice_obj.adata.uns[cent_key] = centrality_df
print(f"\n  Stored in adata.uns['{cent_key}']")
```

## Visualization

### Moran's I — Top Spatially Variable Genes

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

moranI = slice_obj.adata.uns['moranI']
top = moranI.head(20)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(top)), top['I'].values, color='steelblue')
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Moran's I")
ax.set_title("Top Spatially Autocorrelated Genes")
plt.tight_layout()
plt.show()
```

### Co-occurrence Plot

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cluster_key = 'celltype'
co_key = f'{cluster_key}_co_occurrence'
co_results = slice_obj.adata.uns[co_key]
ct_names = slice_obj.adata.obs[cluster_key].cat.categories.tolist()

# Plot co-occurrence at the first interval
occ = co_results['occ']  # shape: (n_types, n_types, n_intervals)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(occ[:, :, 0], cmap='RdBu_r', aspect='auto')
ax.set_xticks(range(len(ct_names)))
ax.set_yticks(range(len(ct_names)))
ax.set_xticklabels(ct_names, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ct_names, fontsize=8)
ax.set_title('Cell Type Co-occurrence')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()
```

---

## Parameter Guide

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `n_neighs` | `10` (cell) / `6` (spot) | 4-30 | Spatial neighbors |
| `cluster_key` | `'celltype'` | Any categorical obs column | For cell-type-level analyses |
| Mode A genes | HVGs | Specific gene list | Genes for Moran's I test |

## Notes

- **Mode A (Moran's I)** answers: "Which genes are spatially patterned?" — good for SVG discovery.
- **Mode B (Ripley's)** answers: "Is this cell type spatially clustered?" — beyond random expectation.
- **Mode C (Co-occurrence)** answers: "Which cell types are found near each other?" — distance-dependent.
- **Mode D (Centrality)** answers: "Which cell type is most central in the tissue architecture?"
- The agent should pick the appropriate mode based on the user's question.
