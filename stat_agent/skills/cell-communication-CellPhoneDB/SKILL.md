---
name: cell-communication-cellphonedb
title: Cell-Cell Communication Analysis (CellPhoneDB)
slug: cell-communication-cellphonedb
description: Analyze cell-cell communication using CellPhoneDB statistical method to identify significant ligand-receptor interactions between cell types via permutation testing.

filter_requirements:
  num_slices: 1
  modalities: [gene]
  data_levels: [cell/spot]

prerequisites:
  - Cell type annotations in target slice (adata.obs['celltype'])
  - Human gene expression data with HGNC gene symbols (CellPhoneDB is human-only), need to confirm currently if it is human data.
default_skill: false
---

# Cell-Cell Communication Analysis Using CellPhoneDB

Identify significant ligand-receptor interactions between cell types using **CellPhoneDB** (v5). CellPhoneDB uses a curated database of human ligand-receptor interactions and statistical permutation testing to determine which interactions are significantly enriched between cell type pairs.

**Important**: CellPhoneDB only supports **human** data. For mouse data, use LIANA+ with `mouseconsensus` resource instead.

**Output**:
- `adata.uns['cellphonedb_means']`: Mean expression of LR pairs per cell type pair
- `adata.uns['cellphonedb_pvalues']`: P-values for each interaction
- `adata.uns['cellphonedb_significant']`: Significant interactions (filtered)
- `adata.uns['cell_communication']`: Summary dictionary

---

## Workflow

### Stage 1: Validation

```python
import numpy as np
import pandas as pd
import scanpy as sc
import os
import tempfile

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

# CellPhoneDB is human-only - verify gene names look like HGNC symbols
sample_genes = adata.var_names[:20].tolist()
has_ensembl = any(g.startswith('ENSG') for g in sample_genes)
if has_ensembl:
    print("  WARNING: Gene names appear to be Ensembl IDs, not HGNC symbols")
    print("  CellPhoneDB requires HGNC symbols (e.g. CD44, EGFR, TNF)")
    print("  Consider converting gene names before running")

n_celltypes = adata.obs['celltype'].nunique()
print(f"✓ Data: {adata.n_obs} cells, {adata.n_vars} genes, {n_celltypes} cell types")
print(f"  Species: human (CellPhoneDB is human-only)")

# Show cell type distribution
print(f"\n  Cell type distribution:")
for ct, count in adata.obs['celltype'].value_counts().head(10).items():
    print(f"    {ct}: {count:,} cells")
```

### Stage 2: Prepare Input Files

```python
print("\n" + "=" * 60)
print("STAGE 2: Prepare Input Files")
print("=" * 60)

# CellPhoneDB uses a file-based API: needs counts TSV + metadata TSV
sample = adata.X[:100]
if hasattr(sample, 'toarray'):
    sample = sample.toarray()
if np.allclose(sample, sample.astype(int)) and sample.max() > 20:
    expr_adata = adata
    counts_source = "adata.X (raw counts)"
    print("  Using adata.X (detected as raw counts)")
else:
    expr_adata = adata
    counts_source = "adata.X (normalized - may reduce sensitivity)"
    print("  WARNING: No raw counts found, using normalized data")
    print("  CellPhoneDB works best with raw counts")

# Create temp directory for CellPhoneDB files
temp_dir = tempfile.mkdtemp(prefix='cpdb_')

# Write metadata file (Cell, cell_type)
meta_df = pd.DataFrame({
    'Cell': adata.obs.index,
    'cell_type': adata.obs['celltype'].astype(str),
})
meta_file = os.path.join(temp_dir, 'meta.tsv')
meta_df.to_csv(meta_file, sep='\t', index=False)
print(f"  ✓ Metadata file written ({len(meta_df)} cells)")

# Write counts file (genes as rows, cells as columns)
counts_file = os.path.join(temp_dir, 'counts.tsv')
X = expr_adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()
expr_matrix = pd.DataFrame(
    X.T,
    index=expr_adata.var_names,
    columns=adata.obs_names,
)
expr_matrix.to_csv(counts_file, sep='\t')
print(f"  ✓ Counts file written ({expr_matrix.shape[0]} genes x {expr_matrix.shape[1]} cells)")
print(f"  Counts source: {counts_source}")

# Download/locate CellPhoneDB database
from cellphonedb.utils import db_utils

cpdb_db_dir = os.path.join(temp_dir, 'db')
os.makedirs(cpdb_db_dir, exist_ok=True)
cpdb_file_path = os.path.join(cpdb_db_dir, 'cellphonedb.zip')

if not os.path.exists(cpdb_file_path):
    print("  Downloading CellPhoneDB v5 database...")
    db_utils.download_database(cpdb_db_dir, "v5.0.0")
    # Find the downloaded file
    for f in os.listdir(cpdb_db_dir):
        if f.endswith('.zip'):
            cpdb_file_path = os.path.join(cpdb_db_dir, f)
            break
    print(f"  ✓ Database downloaded")
else:
    print(f"  ✓ Database found")

print(f"✓ Input files prepared in {temp_dir}")
```

### Stage 3: Run CellPhoneDB Statistical Analysis

```python
print("\n" + "=" * 60)
print("STAGE 3: Run CellPhoneDB Statistical Analysis")
print("=" * 60)

from cellphonedb.src.core.methods import cpdb_statistical_analysis_method

# CellPhoneDB parameters
threshold = 0.1       # Expression threshold (fraction of cells, 0.0-1.0)
iterations = 1000     # Permutations for p-value calculation (more = slower but more accurate)
pvalue_threshold = 0.05  # P-value significance cutoff
result_precision = 3  # Decimal precision for results

output_path = os.path.join(temp_dir, 'output')
os.makedirs(output_path, exist_ok=True)

print(f"  Parameters:")
print(f"    Expression threshold: {threshold}")
print(f"    Permutations: {iterations}")
print(f"    P-value threshold: {pvalue_threshold}")

# Run CellPhoneDB
result = cpdb_statistical_analysis_method.call(
    cpdb_file_path=cpdb_file_path,
    meta_file_path=meta_file,
    counts_file_path=counts_file,
    counts_data='hgnc_symbol',
    threshold=threshold,
    result_precision=result_precision,
    pvalue=pvalue_threshold,
    iterations=iterations,
    debug_seed=42,
    output_path=output_path,
    score_interactions=False,
)

# Validate results
assert isinstance(result, dict), f"Unexpected result type: {type(result)}"
assert 'means' in result, "CellPhoneDB returned no 'means' - check gene name format"

means = result['means']
pvalues = result['pvalues']
significant_means = result['significant_means']
deconvoluted = result['deconvoluted']

# Identify cell type pair columns (format: "CellTypeA|CellTypeB")
meta_cols = ['id_cp_interaction', 'interacting_pair', 'partner_a', 'partner_b',
             'gene_a', 'gene_b', 'secreted', 'receptor_a', 'receptor_b',
             'annotation_strategy', 'is_integrin', 'directionality', 'classification']
pair_cols = [c for c in means.columns if c not in meta_cols]

n_lr_pairs = len(means)
n_cell_type_pairs = len(pair_cols)

print(f"\n✓ CellPhoneDB analysis complete")
print(f"  LR pairs tested: {n_lr_pairs}")
print(f"  Cell type pairs: {n_cell_type_pairs}")
```

### Stage 4: Process Results and Multiple Testing Correction

```python
print("\n" + "=" * 60)
print("STAGE 4: Process Results")
print("=" * 60)

from statsmodels.stats.multitest import multipletests

# Apply FDR correction across cell type pairs for each LR pair
correction_method = 'fdr_bh'  # Options: fdr_bh, bonferroni, sidak, none
pval_array = pvalues[pair_cols].values.astype(float)

n_significant_uncorrected = 0
n_significant_corrected = 0
min_pvals_corrected = np.ones(n_lr_pairs)

for i in range(n_lr_pairs):
    row_pvals = pval_array[i, :]
    # Count uncorrected significant
    n_significant_uncorrected += int((row_pvals < pvalue_threshold).sum())

    if correction_method != 'none':
        valid_mask = ~np.isnan(row_pvals) & (row_pvals >= 0) & (row_pvals <= 1)
        if valid_mask.sum() > 0:
            reject, corrected, _, _ = multipletests(
                row_pvals[valid_mask], alpha=pvalue_threshold, method=correction_method
            )
            n_significant_corrected += int(reject.sum())
            min_pvals_corrected[i] = corrected.min()
    else:
        n_significant_corrected += int((row_pvals < pvalue_threshold).sum())
        min_pvals_corrected[i] = np.nanmin(row_pvals)

# Find significant LR pairs (at least one cell type pair significant)
sig_mask = min_pvals_corrected < pvalue_threshold
n_sig_lr_pairs = int(sig_mask.sum())

print(f"  Correction method: {correction_method}")
print(f"  Significant interactions (uncorrected): {n_significant_uncorrected:,}")
print(f"  Significant interactions (corrected): {n_significant_corrected:,}")
print(f"  Significant LR pairs: {n_sig_lr_pairs}")

# Show top significant interactions
print(f"\n  Top significant interactions:")
sig_means = means[sig_mask].copy()
sig_pvals = pvalues[sig_mask].copy()

if len(sig_means) > 0:
    # Find top interactions by mean expression
    top_count = 0
    for idx, row in sig_means.head(20).iterrows():
        pair_name = row['interacting_pair']
        pval_row = sig_pvals.loc[idx]
        # Find the cell type pair with lowest p-value
        best_col = None
        best_pval = 1.0
        for col in pair_cols:
            p = float(pval_row[col])
            if p < best_pval:
                best_pval = p
                best_col = col
        if best_col and best_pval < pvalue_threshold:
            mean_val = float(row[best_col])
            print(f"    {pair_name} [{best_col}]: mean={mean_val:.3f}, p={best_pval:.4f}")
            top_count += 1
            if top_count >= 15:
                break

# Detect autocrine signaling
autocrine_cols = [c for c in pair_cols if '|' in c and c.split('|')[0] == c.split('|')[1]]
n_autocrine = 0
if autocrine_cols:
    auto_pvals = pvalues[autocrine_cols].values.astype(float)
    n_autocrine = int((auto_pvals < pvalue_threshold).sum())
    print(f"\n  Autocrine signaling: {n_autocrine} significant interactions in {len(autocrine_cols)} self-pairs")
```

### Stage 5: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 5: Storing Results")
print("=" * 60)

# Store results in adata.uns
adata.uns['cellphonedb_means'] = means
adata.uns['cellphonedb_pvalues'] = pvalues
adata.uns['cellphonedb_significant'] = significant_means
adata.uns['cellphonedb_deconvoluted'] = deconvoluted

# Build summary
top_pairs = []
if len(sig_means) > 0:
    # Sort significant LR pairs by their corrected p-value
    sig_order = np.argsort(min_pvals_corrected[sig_mask])[:20]
    for i in sig_order:
        row = sig_means.iloc[i]
        pair_name = row.get('interacting_pair', str(row.name))
        top_pairs.append(pair_name)

summary = {
    'method': 'cellphonedb',
    'species': 'human',
    'n_lr_pairs_tested': n_lr_pairs,
    'n_cell_type_pairs': n_cell_type_pairs,
    'n_significant_lr_pairs': n_sig_lr_pairs,
    'n_significant_interactions': n_significant_corrected,
    'correction_method': correction_method,
    'pvalue_threshold': pvalue_threshold,
    'iterations': iterations,
    'top_interactions': top_pairs,
}
adata.uns['cell_communication'] = summary

# Write results back to session slice
slice_obj.adata.uns['cellphonedb_means'] = adata.uns['cellphonedb_means']
slice_obj.adata.uns['cellphonedb_pvalues'] = adata.uns['cellphonedb_pvalues']
slice_obj.adata.uns['cellphonedb_significant'] = adata.uns['cellphonedb_significant']
slice_obj.adata.uns['cellphonedb_deconvoluted'] = adata.uns['cellphonedb_deconvoluted']
slice_obj.adata.uns['cell_communication'] = adata.uns['cell_communication']

# Clean up temp files
import shutil
shutil.rmtree(temp_dir, ignore_errors=True)

print(f"✓ Results stored in slice_obj.adata")
print(f"  - adata.uns['cellphonedb_means']: Mean expression per interaction per cell type pair")
print(f"  - adata.uns['cellphonedb_pvalues']: P-values per interaction")
print(f"  - adata.uns['cellphonedb_significant']: Significant means (filtered)")
print(f"  - adata.uns['cellphonedb_deconvoluted']: Deconvoluted LR contributions")
print(f"  - adata.uns['cell_communication']: Summary dictionary")

print(f"\n✓ Cell communication analysis complete!")
print(f"  {n_sig_lr_pairs} significant LR pairs found ({n_significant_corrected} interactions total)")
print(f"  Temp files cleaned up")
```

## Visualization

### Heatmap of Significant Interactions

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get significant means matrix
sig_means = adata.uns['cellphonedb_significant']
meta_cols = ['id_cp_interaction', 'interacting_pair', 'partner_a', 'partner_b',
             'gene_a', 'gene_b', 'secreted', 'receptor_a', 'receptor_b',
             'annotation_strategy', 'is_integrin', 'directionality', 'classification']
pair_cols = [c for c in sig_means.columns if c not in meta_cols]

# Filter to rows with at least one nonzero significant value
plot_data = sig_means.set_index('interacting_pair')[pair_cols]
plot_data = plot_data.replace(0, np.nan)
plot_data = plot_data.dropna(how='all')

if len(plot_data) > 30:
    # Keep top 30 by row sum
    plot_data = plot_data.loc[plot_data.sum(axis=1).nlargest(30).index]

fig, ax = plt.subplots(figsize=(max(10, len(pair_cols) * 0.6), max(8, len(plot_data) * 0.3)))
sns.heatmap(
    plot_data.astype(float),
    cmap='YlOrRd',
    ax=ax,
    xticklabels=True,
    yticklabels=True,
    linewidths=0.5,
)
ax.set_title('CellPhoneDB Significant Interactions')
ax.set_xlabel('Cell Type Pairs')
ax.set_ylabel('Ligand-Receptor Pairs')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### Dotplot of Top Interactions

```python
import matplotlib.pyplot as plt
import numpy as np

means_df = adata.uns['cellphonedb_means']
pvals_df = adata.uns['cellphonedb_pvalues']
meta_cols = ['id_cp_interaction', 'interacting_pair', 'partner_a', 'partner_b',
             'gene_a', 'gene_b', 'secreted', 'receptor_a', 'receptor_b',
             'annotation_strategy', 'is_integrin', 'directionality', 'classification']
pair_cols = [c for c in means_df.columns if c not in meta_cols]

# Select top interactions by minimum p-value
min_p = pvals_df[pair_cols].astype(float).min(axis=1)
top_idx = min_p.nsmallest(20).index
top_means = means_df.loc[top_idx].set_index('interacting_pair')[pair_cols].astype(float)
top_pvals = pvals_df.loc[top_idx].set_index('interacting_pair')[pair_cols].astype(float)

fig, ax = plt.subplots(figsize=(max(10, len(pair_cols) * 0.6), max(6, len(top_means) * 0.35)))

for i, (lr_pair, row) in enumerate(top_means.iterrows()):
    for j, col in enumerate(pair_cols):
        mean_val = row[col]
        pval = top_pvals.loc[lr_pair, col]
        if mean_val > 0:
            size = max(5, 50 * (-np.log10(max(pval, 1e-10)) / 10))
            color = 'red' if pval < 0.05 else 'grey'
            ax.scatter(j, i, s=size, c=color, alpha=0.7)

ax.set_xticks(range(len(pair_cols)))
ax.set_xticklabels(pair_cols, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(top_means)))
ax.set_yticklabels(top_means.index, fontsize=8)
ax.set_title('CellPhoneDB: Top LR Interactions (red = p < 0.05)')
plt.tight_layout()
plt.show()
```

---

## Parameter Guide

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `threshold` | `0.1` | 0.0-1.0 | Min fraction of cells expressing gene to consider |
| `iterations` | `1000` | 100-10000 | Permutations for p-value calculation |
| `pvalue_threshold` | `0.05` | 0.0-1.0 | Significance cutoff |
| `correction_method` | `'fdr_bh'` | fdr_bh/bonferroni/sidak/none | Multiple testing correction |
| `result_precision` | `3` | 1-5 | Decimal precision in output |
