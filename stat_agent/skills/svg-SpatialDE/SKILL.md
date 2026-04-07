---
name: svg-spatialde
title: Spatially Variable Genes (SpatialDE)
slug: svg-spatialde
description: Identify spatially variable genes using SpatialDE Gaussian process regression. Decomposes gene expression variance into spatial and non-spatial components to find genes with significant spatial patterns. Uses official NaiveDE preprocessing workflow with Storey q-value FDR correction.

filter_requirements:
  num_slices: 1
  modalities: [gene]
  data_levels: [cell/spot]

prerequisites:
  - No additional prerequisites
default_skill: true
---

# Spatially Variable Genes Using SpatialDE

Identify **spatially variable genes (SVGs)** using **SpatialDE**, which applies Gaussian process regression with spatial kernels to decompose gene expression variance into spatial and non-spatial components. Genes with significant spatial variance are reported as spatially variable.

**What it does**:
- Tests each gene for spatial expression patterns using a likelihood ratio test
- Uses a squared exponential kernel to model spatial covariance
- Returns p-values, FDR-corrected q-values, and spatial length scales per gene

**Output**:
- `adata.var['spatialde_qval']`: FDR-corrected q-values per gene
- `adata.var['spatialde_pval']`: Raw p-values per gene
- `adata.var['spatialde_l']`: Spatial length scale per gene (larger = broader pattern)
- `adata.uns['spatialde_results']`: Full results DataFrame

---

## Workflow

### Stage 1: Load and Validate

```python
import numpy as np
import pandas as pd
import scanpy as sc
import warnings

print("=" * 60)
print("STAGE 1: Load and Validate")
print("=" * 60)

# IMPORTANT: Specify the target slice
slice_id = 0
slice_obj = session.get_slice(slice_id)
adata = slice_obj.adata.copy()

# Ensure unique names
adata.var_names_make_unique()
adata.obs_names_make_unique()

# Validate spatial coordinates
assert 'x' in adata.obs.columns and 'y' in adata.obs.columns, "Missing spatial coordinates"

coords = pd.DataFrame(
    {'x': adata.obs['x'].values, 'y': adata.obs['y'].values},
    index=adata.obs_names,
)

# Determine raw count source
# SpatialDE requires raw counts — check adata.X first, then adata.raw
sample = adata.X[:100]
if hasattr(sample, 'toarray'):
    sample = sample.toarray()

is_raw = np.allclose(sample, sample.astype(int)) and sample.max() > 10

if is_raw:
    raw_X = adata.X
    raw_var_names = adata.var_names
    print(f"  Using adata.X (raw counts detected)")
elif adata.raw is not None:
    raw_X = adata.raw.X
    raw_var_names = adata.raw.var_names
    print(f"  Using adata.raw (adata.X appears normalized)")
else:
    # Fallback: use adata.X and warn
    raw_X = adata.X
    raw_var_names = adata.var_names
    print(f"  WARNING: Data may not be raw counts. SpatialDE expects raw UMI counts.")

print(f"Data: {adata.n_obs} cells/spots, {len(raw_var_names)} genes")
```

### Stage 2: Gene Filtering

```python
print("\n" + "=" * 60)
print("STAGE 2: Gene Filtering")
print("=" * 60)

# SpatialDE is slow on many genes — filter to a manageable set
# IMPORTANT: Adjust n_top_genes based on desired speed vs coverage
n_top_genes = 3000  # Default: 3000 (1000 for quick, 5000+ for thorough)

# Filter genes with very low expression (total counts < 3)
if hasattr(raw_X, 'toarray'):
    gene_totals = np.array(raw_X.sum(axis=0)).flatten()
else:
    gene_totals = np.array(raw_X.sum(axis=0)).flatten()

keep_mask = gene_totals >= 3
filtered_var_names = raw_var_names[keep_mask]
filtered_totals = gene_totals[keep_mask]
print(f"  After low-expression filter (total >= 3): {len(filtered_var_names)} genes")

# Select top genes by total expression (or use HVGs if available)
if len(filtered_var_names) > n_top_genes:
    if 'highly_variable' in adata.var.columns:
        # Prefer HVGs that overlap with filtered genes
        hvg_names = adata.var_names[adata.var['highly_variable']]
        hvg_in_filtered = [g for g in hvg_names if g in filtered_var_names]
        if len(hvg_in_filtered) >= n_top_genes:
            final_genes = hvg_in_filtered[:n_top_genes]
            print(f"  Selected {len(final_genes)} HVGs")
        else:
            # Fill remaining with top-expressed genes
            top_idx = np.argsort(filtered_totals)[::-1][:n_top_genes]
            final_genes = list(filtered_var_names[top_idx])
            print(f"  Selected top {len(final_genes)} genes by expression")
    else:
        top_idx = np.argsort(filtered_totals)[::-1][:n_top_genes]
        final_genes = list(filtered_var_names[top_idx])
        print(f"  Selected top {len(final_genes)} genes by expression")
else:
    final_genes = list(filtered_var_names)
    print(f"  Using all {len(final_genes)} filtered genes")

# Build dense count matrix for selected genes
gene_indices = [list(raw_var_names).index(g) for g in final_genes]

if hasattr(raw_X, 'toarray'):
    counts = pd.DataFrame(
        raw_X[:, gene_indices].toarray(),
        columns=final_genes,
        index=adata.obs_names,
    )
else:
    counts = pd.DataFrame(
        raw_X[:, gene_indices],
        columns=final_genes,
        index=adata.obs_names,
    )

counts = counts.astype(np.float64)
print(f"  Count matrix: {counts.shape}")
```

### Stage 3: SpatialDE Preprocessing

```python
print("\n" + "=" * 60)
print("STAGE 3: SpatialDE Preprocessing")
print("=" * 60)

# Apply scipy compatibility patches before importing SpatialDE
import scipy

# Patch 1: SpatialDE/util.py uses sp.arange and sp.array which were removed in scipy >= 1.14
# These were always re-exports from numpy, so shimming them back is safe
if not hasattr(scipy, 'arange'):
    scipy.arange = np.arange
if not hasattr(scipy, 'array'):
    scipy.array = np.array
if not hasattr(scipy, 'argsort'):
    scipy.argsort = np.argsort
if not hasattr(scipy, 'zeros_like'):
    scipy.zeros_like = np.zeros_like

# Patch 2: scipy >= 1.14 removed scipy.misc.derivative
try:
    import scipy.misc
    if not hasattr(scipy.misc, 'derivative'):
        from scipy.misc import derivative
except (ImportError, AttributeError):
    pass

import NaiveDE
import SpatialDE

# Step 1: Variance stabilization (delta-method, assumes log-normal)
print("  Applying variance stabilization (NaiveDE.stabilize)...")
norm_expr = NaiveDE.stabilize(counts.T).T

# Step 2: Regress out library size effects
print("  Regressing out library size (NaiveDE.regress_out)...")
total_counts = pd.DataFrame(
    {'total_counts': counts.sum(axis=1)},
    index=counts.index,
)

# Replace zero total counts to avoid log(0)
total_counts['total_counts'] = total_counts['total_counts'].replace(0, 1)

resid_expr = NaiveDE.regress_out(
    total_counts, norm_expr.T, 'np.log(total_counts)'
).T

# Handle any NaN/Inf from preprocessing
resid_expr = resid_expr.fillna(0)
resid_expr = resid_expr.replace([np.inf, -np.inf], 0)

print(f"  Preprocessed expression matrix: {resid_expr.shape}")
```

### Stage 4: Run SpatialDE

```python
print("\n" + "=" * 60)
print("STAGE 4: Run SpatialDE")
print("=" * 60)

n_genes = resid_expr.shape[1]
print(f"  Testing {n_genes} genes for spatial variability...")
if n_genes > 5000:
    print(f"  NOTE: Testing >5000 genes — this may take a while")

# Run SpatialDE (Gaussian process regression with squared exponential kernel)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results = SpatialDE.run(coords.values, resid_expr)

# Multiple testing correction using Storey q-value method
from SpatialDE.util import qvalue
results['qval'] = qvalue(results['pval'].values)

# Sort by q-value
results = results.sort_values('qval').reset_index(drop=True)

# Count significant genes
n_significant = (results['qval'] < 0.05).sum()
print(f"SpatialDE complete")
print(f"  Genes tested: {len(results)}")
print(f"  Significant SVGs (q < 0.05): {n_significant}")

# Show top SVGs
print(f"\n  Top 20 spatially variable genes:")
for _, row in results.head(20).iterrows():
    print(f"    {row['g']}: q={row['qval']:.2e}, p={row['pval']:.2e}, length_scale={row['l']:.1f}")
```

### Stage 5: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 5: Store Results")
print("=" * 60)

# Map results back to adata.var by gene name
results_indexed = results.set_index('g')

# Store per-gene statistics in adata.var
for col, var_col in [('pval', 'spatialde_pval'), ('qval', 'spatialde_qval'), ('l', 'spatialde_l')]:
    values = pd.Series(np.nan, index=slice_obj.adata.var_names)
    overlap = values.index.intersection(results_indexed.index)
    values.loc[overlap] = results_indexed.loc[overlap, col].values
    slice_obj.adata.var[var_col] = values.values

# Store full results DataFrame
slice_obj.adata.uns['spatialde_results'] = results

# Store list of significant SVGs for easy access
svg_list = results[results['qval'] < 0.05]['g'].tolist()
slice_obj.adata.uns['spatially_variable_genes'] = svg_list

print(f"Added 'spatialde_pval' to slice_obj.adata.var")
print(f"Added 'spatialde_qval' to slice_obj.adata.var")
print(f"Added 'spatialde_l' (length scale) to slice_obj.adata.var")
print(f"Added 'spatialde_results' (full DataFrame) to slice_obj.adata.uns")
print(f"Added 'spatially_variable_genes' ({len(svg_list)} genes) to slice_obj.adata.uns")
print(f"\nSpatialDE analysis complete!")
```

## Visualization

### Top SVGs Expression Heatmap

```python
import matplotlib.pyplot as plt

adata = slice_obj.adata
results = adata.uns['spatialde_results']
top_genes = results.head(6)['g'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for ax, gene in zip(axes.ravel(), top_genes):
    if gene in adata.var_names:
        expr = adata[:, gene].X
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        expr = expr.flatten()

        sc_plot = ax.scatter(
            adata.obs['x'], adata.obs['y'],
            c=expr, cmap='viridis', s=3, alpha=0.8,
        )
        qval = adata.var.loc[gene, 'spatialde_qval']
        ax.set_title(f'{gene} (q={qval:.2e})')
        plt.colorbar(sc_plot, ax=ax, shrink=0.7)

plt.suptitle('Top Spatially Variable Genes (SpatialDE)')
plt.tight_layout()
plt.show()
```

### Volcano-style Plot (Spatial Fraction vs Significance)

```python
import matplotlib.pyplot as plt

results = slice_obj.adata.uns['spatialde_results']

fig, ax = plt.subplots(figsize=(8, 6))
sig = results['qval'] < 0.05
ax.scatter(results.loc[~sig, 'l'], -np.log10(results.loc[~sig, 'qval']),
           c='gray', s=5, alpha=0.5, label='Not significant')
ax.scatter(results.loc[sig, 'l'], -np.log10(results.loc[sig, 'qval']),
           c='red', s=8, alpha=0.7, label=f'Significant (n={sig.sum()})')

ax.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Spatial length scale')
ax.set_ylabel('-log10(q-value)')
ax.set_title('SpatialDE Results')
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Parameter Guide

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_top_genes` | `3000` | 500-10000 | Number of genes to test. More = thorough but slower |

## Notes

- **Raw counts required**: SpatialDE performs its own preprocessing (NaiveDE variance stabilization + library size regression). Do NOT pass normalized or log-transformed data. The skill automatically checks `adata.X` and falls back to `adata.raw`.
- **Performance**: SpatialDE tests ~1000 genes in a few minutes. At 5000+ genes it becomes slow. Default 3000 is a good balance. Use HVGs or top-expressed genes to reduce the set.
- **Length scale (`l`)**: The spatial length scale indicates the range of the spatial pattern. Larger values = broader/smoother patterns, smaller values = local/fine-grained patterns.
- **Q-value threshold**: q < 0.05 is the standard significance cutoff. The Storey q-value method adaptively estimates the proportion of true nulls (pi0).
- **NaiveDE dependency**: SpatialDE requires the `NaiveDE` module (installed alongside SpatialDE) for preprocessing. The official workflow is: filter → stabilize → regress_out → SpatialDE.run.
- **Scipy compatibility**: SpatialDE's `util.py` uses `scipy.arange`/`scipy.array` (removed in scipy >= 1.14) and `scipy.misc.derivative`. The skill patches these at runtime by shimming the numpy equivalents back onto scipy before importing SpatialDE.
- **Comparison with SPARK-X**: SpatialDE uses Gaussian process regression (parametric, slower). SPARK-X uses non-parametric kernel tests (faster, more scalable). SpatialDE provides length scales; SPARK-X does not.
