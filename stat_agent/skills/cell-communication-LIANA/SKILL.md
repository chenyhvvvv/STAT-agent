---
name: cell-communication-liana
title: Cell-Cell Communication Analysis (LIANA+)
slug: cell-communication-liana
description: Analyze cell-cell communication using LIANA+ to identify significant ligand-receptor interactions between cell types. (Recommended!)
filter_requirements:
  num_slices: 1
  modalities: [gene]
  data_levels: [cell/spot]

prerequisites:
  - Cell type annotations in target slice (adata.obs['celltype'])
  - Species information (human or mouse) for selecting the correct ligand-receptor database
default_skill: true
---

# Cell-Cell Communication Analysis Using LIANA+

Identify significant ligand-receptor (LR) interactions between cell types using **LIANA+** (LIgand-receptor ANAlysis framework). LIANA+ aggregates multiple scoring methods (CellPhoneDB, NATMI, SingleCellSignalR, etc.) into a robust consensus ranking.

Identifies which cell type pairs communicate via which LR pairs (aggregate across all cells per type)

**Output**:
- `adata.uns['liana_res']`: Full cluster-level results DataFrame
- `adata.uns['cell_communication']`: Summary dictionary

---

## Workflow

### Stage 1: Validation

```python
import liana as li
import numpy as np
import pandas as pd
import scanpy as sc

# IMPORTANT: Specify the target slice
slice_id = 0
slice_obj = session.get_slice(slice_id)
adata = slice_obj.adata.copy()

print("=" * 60)
print("STAGE 1: Validation")
print("=" * 60)

# Validate cell type annotations
assert 'celltype' in adata.obs.columns, "Missing 'celltype' column - run cell type annotation first"
assert adata.obs['celltype'].notna().all(), "celltype column contains NaN values"

# Validate spatial coordinates
assert 'x' in adata.obs.columns and 'y' in adata.obs.columns, "Missing spatial coordinates"

# Ensure unique names
adata.var_names_make_unique()
adata.obs_names_make_unique()

# Species and LR database configuration
species = 'human'  # IMPORTANT: Set to 'human' or 'mouse' based on the dataset
resource_name = 'consensus'  # Default: 'consensus' (human) or 'mouseconsensus' (mouse)

# Auto-select correct resource for mouse
if species == 'mouse' and resource_name == 'consensus':
    resource_name = 'mouseconsensus'
    print("  Auto-selected 'mouseconsensus' resource for mouse data")

n_celltypes = adata.obs['celltype'].nunique()
print(f"✓ Data: {adata.n_obs} cells, {adata.n_vars} genes, {n_celltypes} cell types")
print(f"  Species: {species}")
print(f"  LR database: {resource_name}")

# Show cell type distribution
print(f"\n  Cell type distribution:")
for ct, count in adata.obs['celltype'].value_counts().head(10).items():
    print(f"    {ct}: {count:,} cells")
```

### Stage 2: Preprocessing

```python
print("\n" + "=" * 60)
print("STAGE 2: Preprocessing")
print("=" * 60)

# Create spatial coordinates array
adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()

# Check if data needs normalization
# LIANA expects log-normalized data in adata.X; raw counts can go in adata.raw
use_raw = False
if adata.raw is not None:
    use_raw = True
    print("  Found adata.raw, will use raw counts for cluster analysis")
else:
    # Detect raw counts: integer values with large max
    sample = adata.X[:100]
    if hasattr(sample, 'toarray'):
        sample = sample.toarray()
    if np.allclose(sample, sample.astype(int)) and sample.max() > 20:
        adata.raw = adata.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        use_raw = True
        print("  Detected raw counts -> normalized and log-transformed")
        print("  Raw counts stored in adata.raw")
    else:
        print("  Data appears already normalized")

# Compute spatial neighbors (required for spatial bivariate analysis)
print("  Computing spatial neighbors...")
try:
    import squidpy as sq
    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        n_neighs=min(30, max(6, adata.n_obs // 100)),
        delaunay=True,
        set_diag=False,
    )
    print(f"  ✓ Spatial neighbors computed via squidpy")
except ImportError:
    # Fallback: manual KNN using scipy
    from scipy.spatial import KDTree
    from scipy.sparse import csr_matrix

    coords = adata.obsm['spatial']
    n_neighbors = min(30, max(6, adata.n_obs // 100))
    tree = KDTree(coords)
    distances, indices = tree.query(coords, k=n_neighbors + 1)

    rows, cols, vals = [], [], []
    for i in range(len(coords)):
        for j_idx in range(1, n_neighbors + 1):
            j = indices[i, j_idx]
            rows.append(i)
            cols.append(j)
            vals.append(1.0)

    connectivity = csr_matrix((vals, (rows, cols)), shape=(len(coords), len(coords)))
    adata.obsp['spatial_connectivities'] = connectivity
    print(f"  ✓ Spatial neighbors computed via KDTree (n_neighbors={n_neighbors})")

print(f"✓ Preprocessing complete")
```

### Stage 3: Cluster-Based Communication Analysis

```python
print("\n" + "=" * 60)
print("STAGE 3: Cluster-Based Communication Analysis")
print("=" * 60)

# Run LIANA+ rank_aggregate (consensus of multiple scoring methods)
li.mt.rank_aggregate(
    adata,
    groupby='celltype',
    resource_name=resource_name,
    expr_prop=0.1,       # Min proportion of cells expressing the gene (0.0-1.0)
    min_cells=3,         # Min cells per cell type expressing ligand or receptor
    n_perms=1000,        # Permutations for p-value calculation
    verbose=False,
    use_raw=use_raw,
)

# Extract results
liana_res = adata.uns["liana_res"]
n_total = len(liana_res)

# Count significant interactions (magnitude_rank <= 0.05)
significance_alpha = 0.05
n_significant = int((liana_res["magnitude_rank"] <= significance_alpha).sum())

print(f"✓ Cluster analysis complete")
print(f"  Total LR interactions tested: {n_total:,}")
print(f"  Significant interactions (rank <= {significance_alpha}): {n_significant:,}")

# Show top interactions
print(f"\n  Top 15 ligand-receptor interactions:")
top_interactions = liana_res.nsmallest(15, "magnitude_rank")
for _, row in top_interactions.iterrows():
    print(f"    {row['source']} -> {row['target']}: "
          f"{row['ligand_complex']}-{row['receptor_complex']} "
          f"(rank: {row['magnitude_rank']:.4f})")

# Detect autocrine signaling (source == target cell type)
autocrine = liana_res[liana_res['source'] == liana_res['target']]
if len(autocrine) > 0:
    n_autocrine_sig = int((autocrine['magnitude_rank'] <= significance_alpha).sum())
    print(f"\n  Autocrine signaling: {len(autocrine)} pairs ({n_autocrine_sig} significant)")
```



### Stage 5: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 5: Storing Results")
print("=" * 60)

# Build summary
summary = {
    'method': 'liana',
    'species': species,
    'resource': resource_name,
    'n_interactions': n_total,
    'n_significant': n_significant,
    'significance_threshold': significance_alpha,
    'top_interactions': [
        f"{row['source']}->{row['target']}: {row['ligand_complex']}-{row['receptor_complex']}"
        for _, row in liana_res.nsmallest(20, 'magnitude_rank').iterrows()
    ],
}

# Add spatial results if Stage 4 was run
try:
    summary['n_spatial_significant'] = n_spatial_significant
    summary['spatial_local_metric'] = local_metric
    summary['spatial_global_metric'] = global_metric
except NameError:
    pass  # Stage 4 was skipped

adata.uns['cell_communication'] = summary

# Write results back to session slice
slice_obj.adata.uns['liana_res'] = adata.uns['liana_res']
slice_obj.adata.uns['cell_communication'] = adata.uns['cell_communication']

if 'spatial_connectivities' in adata.obsp:
    slice_obj.adata.obsp['spatial_connectivities'] = adata.obsp['spatial_connectivities']

print(f"✓ Results stored in slice_obj.adata")
print(f"  - adata.uns['liana_res']: Full cluster analysis results DataFrame")
print(f"  - adata.uns['cell_communication']: Summary dictionary")

print(f"\n✓ Cell communication analysis complete!")
print(f"  {n_significant} significant cluster-level interactions found")
```

## Visualization

### Dotplot of Top Interactions

```python
import plotnine as p9 # For adjust LIANA+ built-in dotplot
# Size = significance (inverse magnitude_rank), Color = expression (lr_means)
# Visualization: Dotplot of Top Interactions
# Size = significance (inverse magnitude_rank), Color = expression (lr_means)
plot = li.pl.dotplot(
    adata=adata,
    colour='magnitude_rank',
    size='lr_means',
    inverse_size=True,        # Smaller rank = larger dot
    top_n=20,
    orderby='magnitude_rank',
    orderby_ascending=True,
    size_range=(0.5, 1)
)

(    
    plot
    + p9.theme_bw(base_size=6)
    + p9.theme(axis_text_x=p9.element_text(angle=90), figure_size=(24, 8))
).show()

```

---

## Parameter Guide

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `resource_name` | `'consensus'` | See below | Ligand-receptor database |
| `expr_prop` | `0.1` | 0.0-1.0 | Min proportion of cells expressing the gene |
| `min_cells` | `3` | 1-100 | Min cells per type expressing ligand or receptor |
| `n_perms` | `1000` | 100-10000 | Permutations (more = slower but more accurate p-values) |
| `local_metric` | `'cosine'` | cosine/pearson/spearman/jaccard | Spatial local metric |
| `global_metric` | `'morans'` | morans/lee | Spatial global statistic |

**Available LR databases:**
- `consensus` (default, human) - Consensus of multiple databases (recommended)
- `mouseconsensus` (mouse) - Mouse-specific consensus
- `cellphonedb` - CellPhoneDB (stringent, human-only)
- `cellchatdb` - CellChat database
- `celltalkdb` - CellTalkDB (large coverage)
- `connectomedb2020` - Connectome DB 2020
- `icellnet` - iCellNet (immune-focused)
