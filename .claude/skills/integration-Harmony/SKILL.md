---
name: integration-harmony
title: Batch Integration (Harmony)
slug: integration-harmony
description: Integrate multiple spatial transcriptomics slices using Harmony batch correction. User specifies which slices to integrate and an optional reference scRNA-seq with cell type annotations. Harmony corrects batch effects in PCA space and stores corrected embeddings back into each slice. If slices lack cell type annotations and a reference is provided, KNN label transfer can annotate cells using the integrated embedding.

filter_requirements:
  modalities: [gene]
  data_levels: [cell/spot]

prerequisites:
  - Which slice IDs to integrate (at least 2 gene-expression slices)
  - Optional: path to a reference scRNA-seq h5ad file with celltype annotations
  - Optional: whether to use KNN label transfer from reference (if slices lack celltypes)
---

# Batch Integration Using Harmony

Integrate **user-specified spatial slices** using **Harmony** batch correction. Optionally include an external scRNA-seq reference with cell type annotations for KNN-based label transfer.

**Input**:
- `slice_ids`: Which slices to integrate (user must specify, e.g., `[0, 1]`)
- `reference_path`: Optional path to a reference scRNA-seq h5ad with `celltype` column
- `use_knn_transfer`: If True and reference is provided, transfer cell type labels to slices without annotations via KNN in the Harmony-corrected embedding space

**What it does**:
1. Concatenates specified slices (+ optional reference) with batch labels
2. Preprocesses, selects HVGs with batch correction, computes PCA
3. Runs Harmony to correct batch effects in PCA space
4. Computes UMAP before and after integration for comparison
5. Optionally transfers cell type labels from reference via KNN
6. Stores embeddings back into **each slice's adata** (no new slice created)

**Results stored in each slice's adata**:
- `adata.obsm['X_pca_harmony']`: Batch-corrected PCA embedding
- `adata.obsm['X_umap_pre_harmony']`: UMAP before integration (for comparison)
- `adata.obsm['X_umap_harmony']`: UMAP after integration
- `adata.obs['harmony_batch']`: Batch label used during integration
- `adata.obs['celltype']`: Cell type labels (if KNN transfer was performed)
- `adata.uns['harmony_params']`: Integration parameters for provenance

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

# IMPORTANT: User must specify which slices to integrate
slice_ids = [0, 1]  # <-- MODIFY: specify slice IDs to integrate

# Optional: path to reference scRNA-seq h5ad with celltype annotations
reference_path = None  # <-- MODIFY: e.g., '/path/to/reference.h5ad'

# Optional: transfer cell type labels from reference via KNN
use_knn_transfer = False  # <-- MODIFY: set True if slices lack celltypes and reference is provided

# Validate slices
assert len(slice_ids) >= 2, f"Need at least 2 slices, got {len(slice_ids)}"

adatas = []
batch_labels = []
slice_has_celltype = {}

for sid in slice_ids:
    slice_obj = session.get_slice(sid)
    adata_i = slice_obj.adata.copy()
    tissue_name = slice_obj.metadata.get('tissue_name', f'slice_{sid}')
    adata_i.obs['batch'] = tissue_name
    adata_i.obs['_source'] = 'query'
    adata_i.obs['_slice_id'] = sid
    has_ct = 'celltype' in adata_i.obs.columns and adata_i.obs['celltype'].notna().any()
    slice_has_celltype[sid] = has_ct
    adatas.append(adata_i)
    batch_labels.append(tissue_name)
    ct_status = f", celltypes: {adata_i.obs['celltype'].nunique()} types" if has_ct else ", no celltypes"
    print(f"  Slice {sid} ({tissue_name}): {adata_i.n_obs} cells/spots, {adata_i.n_vars} genes{ct_status}")

# Load optional reference
ref_adata = None
if reference_path is not None:
    import anndata as ad
    ref_adata = ad.read_h5ad(reference_path)
    assert 'celltype' in ref_adata.obs.columns, "Reference must have 'celltype' column in obs"
    ref_adata.obs['batch'] = 'reference'
    ref_adata.obs['_source'] = 'reference'
    ref_adata.obs['_slice_id'] = -1
    n_ref_types = ref_adata.obs['celltype'].nunique()
    print(f"  Reference: {ref_adata.n_obs} cells, {ref_adata.n_vars} genes, {n_ref_types} cell types")
    batch_labels.append('reference')

any_slice_has_celltype = any(slice_has_celltype.values())

# Auto-enable KNN if reference provided and slices lack celltypes
if ref_adata is not None and not any_slice_has_celltype and not use_knn_transfer:
    print("\n  NOTE: No slices have celltype annotations but reference is provided.")
    print("  Enabling KNN label transfer automatically.")
    use_knn_transfer = True

if use_knn_transfer and ref_adata is None:
    print("\n  WARNING: KNN transfer requested but no reference provided. Skipping KNN.")
    use_knn_transfer = False

print(f"\nSlices to integrate: {slice_ids}")
print(f"Reference: {'yes' if ref_adata is not None else 'no'}")
print(f"KNN label transfer: {'yes' if use_knn_transfer else 'no'}")
```

### Stage 2: Concatenate and Preprocess

```python
print("\n" + "=" * 60)
print("STAGE 2: Concatenate and Preprocess")
print("=" * 60)

# Prepare all datasets for concatenation
all_adatas = adatas.copy()
if ref_adata is not None:
    all_adatas.append(ref_adata)

# Ensure unique names
for a in all_adatas:
    a.var_names_make_unique()
    a.obs_names_make_unique()

# Concatenate
combined = all_adatas[0].concatenate(
    all_adatas[1:],
    batch_key='_concat_batch',
    join='inner',
)

if '_concat_batch' in combined.obs.columns:
    combined.obs.drop(columns=['_concat_batch'], inplace=True)

# Clean var columns with NA from concatenation
for col in combined.var.columns:
    if combined.var[col].dtype == 'object' and combined.var[col].isna().any():
        unique_vals = combined.var[col].dropna().unique()
        if set(unique_vals).issubset({True, False, 'True', 'False'}):
            combined.var[col] = combined.var[col].fillna(False).astype(bool)
        else:
            combined.var[col] = combined.var[col].fillna('').astype(str)

# Remove artifacts from concatenation
for key in ['X_diffmap', 'X_pca', 'X_umap']:
    if key in combined.obsm:
        del combined.obsm[key]
if 'diffmap_evals' in combined.uns:
    del combined.uns['diffmap_evals']

print(f"  Combined: {combined.n_obs} cells/spots, {combined.n_vars} shared genes")
print(f"  Batches: {combined.obs['batch'].value_counts().to_dict()}")

# Check if data needs preprocessing
sample = combined.X[:100]
if hasattr(sample, 'toarray'):
    sample = sample.toarray()
is_raw = np.allclose(sample, sample.astype(int)) and sample.max() > 10

if is_raw:
    print("  Data appears to be raw counts. Normalizing...")
    combined.raw = combined.copy()
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    print("  Applied: normalize_total(target_sum=1e4) + log1p")
else:
    max_val = sample.max()
    if max_val < 20:
        print(f"  Data appears already normalized (max={max_val:.1f})")
    else:
        print(f"  High values detected (max={max_val:.1f}). Normalizing...")
        combined.raw = combined.copy()
        sc.pp.normalize_total(combined, target_sum=1e4)
        sc.pp.log1p(combined)
```

### Stage 3: HVG Selection and PCA

```python
print("\n" + "=" * 60)
print("STAGE 3: HVG Selection and PCA")
print("=" * 60)

# IMPORTANT: Parameters
n_top_genes = 2000
n_pcs = 30

sc.pp.highly_variable_genes(
    combined,
    n_top_genes=n_top_genes,
    batch_key='batch',
)

n_hvg = combined.var['highly_variable'].sum()
print(f"  Highly variable genes: {n_hvg}")

combined_hvg = combined[:, combined.var['highly_variable']].copy()
sc.pp.scale(combined_hvg, zero_center=True, max_value=10)

max_pcs = min(n_pcs, combined_hvg.n_vars - 1, combined_hvg.n_obs - 1)
sc.tl.pca(combined_hvg, n_comps=max_pcs, svd_solver='arpack')

print(f"  PCA: {max_pcs} components")
print(f"  Variance explained (first 5): {combined_hvg.uns['pca']['variance_ratio'][:5].round(3)}")

# Compute UMAP BEFORE integration (for comparison)
sc.pp.neighbors(combined_hvg, use_rep='X_pca', n_neighbors=15)
sc.tl.umap(combined_hvg)
X_umap_before = combined_hvg.obsm['X_umap'].copy()
print("  Pre-integration UMAP computed")
```

### Stage 4: Run Harmony

```python
print("\n" + "=" * 60)
print("STAGE 4: Run Harmony")
print("=" * 60)

import harmonypy

# IMPORTANT: Parameters
max_iter_harmony = 20

X_pca = combined_hvg.obsm['X_pca']
n_cells = combined_hvg.n_obs
meta_data = pd.DataFrame({'batch': combined_hvg.obs['batch'].values})

print(f"  Running Harmony on {n_cells} cells, {X_pca.shape[1]} PCs, {combined_hvg.obs['batch'].nunique()} batches...")

harmony_out = harmonypy.run_harmony(
    data_mat=X_pca,
    meta_data=meta_data,
    vars_use=['batch'],
    max_iter_harmony=max_iter_harmony,
    verbose=True,
)

    # Handle harmonypy version compatibility
    Z_corr = harmony_out.Z_corr
    if Z_corr.shape[0] == n_cells:
        combined_hvg.obsm['X_pca_harmony'] = Z_corr
    else:
        combined_hvg.obsm['X_pca_harmony'] = Z_corr.T

    print(f"  Harmony complete. Corrected embedding: {combined_hvg.obsm['X_pca_harmony'].shape}")

    # Compute UMAP AFTER integration
    sc.pp.neighbors(combined_hvg, use_rep='X_pca_harmony', n_neighbors=15)
    sc.tl.umap(combined_hvg)
    X_umap_after = combined_hvg.obsm['X_umap'].copy()
    print("  Post-integration UMAP computed")
```

### Stage 5: KNN Label Transfer (Optional)

```python
print("\n" + "=" * 60)
print("STAGE 5: KNN Label Transfer")
print("=" * 60)

if use_knn_transfer and ref_adata is not None:
    from sklearn.neighbors import KNeighborsClassifier

    # IMPORTANT: Parameters
    n_knn = 15  # Number of neighbors for KNN

    # Split combined into reference and query cells
    ref_mask = combined_hvg.obs['_source'] == 'reference'
    query_mask = combined_hvg.obs['_source'] == 'query'

    X_harmony = combined_hvg.obsm['X_pca_harmony']
    X_ref = X_harmony[ref_mask]
    X_query = X_harmony[query_mask]
    y_ref = combined_hvg.obs.loc[ref_mask, 'celltype'].values

    print(f"  Reference: {X_ref.shape[0]} cells, {len(np.unique(y_ref))} cell types")
    print(f"  Query: {X_query.shape[0]} cells")

    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_knn, weights='distance', metric='euclidean')
    knn.fit(X_ref, y_ref)

    # Predict cell types for query cells
    predicted_celltypes = knn.predict(X_query)
    knn_proba = knn.predict_proba(X_query)
    knn_confidence = np.max(knn_proba, axis=1)

    # Store predictions in combined object (query cells only)
    combined_hvg.obs.loc[query_mask, 'celltype_knn'] = predicted_celltypes
    combined_hvg.obs.loc[query_mask, 'celltype_knn_confidence'] = knn_confidence

    # Summary
    print(f"\n  KNN transfer results:")
    for ct in sorted(np.unique(predicted_celltypes)):
        n = np.sum(predicted_celltypes == ct)
        mean_conf = knn_confidence[predicted_celltypes == ct].mean()
        print(f"    {ct}: {n} cells (mean confidence: {mean_conf:.3f})")
    print(f"  Overall mean confidence: {knn_confidence.mean():.3f}")
else:
    if not use_knn_transfer:
        print("  KNN label transfer: skipped (not requested)")
    else:
        print("  KNN label transfer: skipped (no reference)")
```

### Stage 6: Store Results Back to Slices

```python
print("\n" + "=" * 60)
print("STAGE 6: Store Results Back to Slices")
print("=" * 60)

# Build index mapping: for each cell in combined, which slice does it belong to?
query_mask = combined_hvg.obs['_source'] == 'query'
query_obs = combined_hvg.obs[query_mask]

for sid in slice_ids:
    slice_obj = session.get_slice(sid)
    slice_mask = query_obs['_slice_id'] == sid
    idx = np.where(query_mask)[0][slice_mask.values]

    # Store harmony embedding
    slice_obj.adata.obsm['X_pca_harmony'] = combined_hvg.obsm['X_pca_harmony'][idx]

    # Store UMAP before and after integration
    slice_obj.adata.obsm['X_umap_pre_harmony'] = X_umap_before[idx]
    slice_obj.adata.obsm['X_umap_harmony'] = X_umap_after[idx]

    # Store batch label
    slice_obj.adata.obs['harmony_batch'] = combined_hvg.obs['batch'].values[idx]

    # Store KNN-transferred celltypes if applicable
    if use_knn_transfer and 'celltype_knn' in combined_hvg.obs.columns:
        if not slice_has_celltype[sid]:
            slice_obj.adata.obs['celltype'] = combined_hvg.obs['celltype_knn'].values[idx]
            slice_obj.adata.obs['celltype_knn_confidence'] = combined_hvg.obs['celltype_knn_confidence'].values[idx]
            print(f"  Slice {sid}: stored harmony embeddings + KNN celltypes ({slice_obj.adata.obs['celltype'].nunique()} types)")
        else:
            # Slice already has celltypes; store KNN result separately for comparison
            slice_obj.adata.obs['celltype_knn'] = combined_hvg.obs['celltype_knn'].values[idx]
            slice_obj.adata.obs['celltype_knn_confidence'] = combined_hvg.obs['celltype_knn_confidence'].values[idx]
            print(f"  Slice {sid}: stored harmony embeddings (kept original celltypes)")
    else:
        print(f"  Slice {sid}: stored harmony embeddings")

    # Store integration parameters
    slice_obj.adata.uns['harmony_params'] = {
        'integrated_slices': slice_ids,
        'batch_labels': batch_labels,
        'n_top_genes': n_top_genes,
        'n_pcs': max_pcs,
        'max_iter_harmony': max_iter_harmony,
        'has_reference': ref_adata is not None,
        'knn_transfer': use_knn_transfer,
    }

print(f"\nHarmony integration complete!")
print(f"  Embeddings stored: X_pca_harmony, X_umap_pre_harmony, X_umap_harmony")
if use_knn_transfer:
    print(f"  KNN cell types stored in obs['celltype'] for unlabeled slices")
```

## Visualization

### Before vs After Integration (UMAP by Batch)

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

# Use the combined data for visualization (has both before/after UMAP)
query_mask_vis = combined_hvg.obs['_source'] == 'query'
batches = combined_hvg.obs.loc[query_mask_vis, 'batch'].unique()

# Include reference in batch list if present
all_sources = combined_hvg.obs['_source'].unique()
include_ref = 'reference' in all_sources
if include_ref:
    all_batches = list(batches) + ['reference']
else:
    all_batches = list(batches)

colors_map = plt.cm.Set2(np.linspace(0, 1, len(all_batches)))
batch_color_map = dict(zip(all_batches, colors_map))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Panel 1: UMAP BEFORE integration colored by batch ---
for batch in all_batches:
    mask = combined_hvg.obs['batch'] == batch
    axes[0].scatter(
        X_umap_before[mask, 0], X_umap_before[mask, 1],
        c=[batch_color_map[batch]], s=2, alpha=0.5, label=batch,
    )
axes[0].legend(markerscale=5, framealpha=0.9)
axes[0].set_title('Before Harmony (by batch)')
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')

# --- Panel 2: UMAP AFTER integration colored by batch ---
for batch in all_batches:
    mask = combined_hvg.obs['batch'] == batch
    axes[1].scatter(
        X_umap_after[mask, 0], X_umap_after[mask, 1],
        c=[batch_color_map[batch]], s=2, alpha=0.5, label=batch,
    )
axes[1].legend(markerscale=5, framealpha=0.9)
axes[1].set_title('After Harmony (by batch)')
axes[1].set_xlabel('UMAP1')
axes[1].set_ylabel('UMAP2')

plt.suptitle('Harmony Integration: Batch Effect Correction', fontsize=14)
plt.tight_layout()
plt.show()
```

### UMAP Colored by Cell Type (Per Slice)

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Determine which cell type column to use for visualization
has_any_celltype_for_vis = any(slice_has_celltype.values())
has_knn_celltypes = use_knn_transfer and 'celltype_knn' in combined_hvg.obs.columns

# Collect all cell type labels for a unified color map
all_celltypes_set = set()
if has_any_celltype_for_vis or has_knn_celltypes:
    # From slices with original celltypes
    for sid in slice_ids:
        if slice_has_celltype[sid] and 'celltype' in combined_hvg.obs.columns:
            mask = (combined_hvg.obs['_slice_id'] == sid) & (combined_hvg.obs['_source'] == 'query')
            cts = combined_hvg.obs.loc[mask, 'celltype'].dropna().unique()
            all_celltypes_set.update(cts)
    # From KNN predictions
    if has_knn_celltypes:
        cts = combined_hvg.obs['celltype_knn'].dropna().unique()
        all_celltypes_set.update(cts)
    # From reference
    if include_ref and 'celltype' in combined_hvg.obs.columns:
        ref_mask = combined_hvg.obs['_source'] == 'reference'
        cts = combined_hvg.obs.loc[ref_mask, 'celltype'].dropna().unique()
        all_celltypes_set.update(cts)

if len(all_celltypes_set) > 0:
    all_celltypes_list = sorted(all_celltypes_set)
    ct_colors = plt.cm.tab20(np.linspace(0, 1, max(len(all_celltypes_list), 1)))
    ct_color_map = dict(zip(all_celltypes_list, ct_colors))

    # Count panels: one per slice with celltypes + one for reference if present
    panels = []
    for sid in slice_ids:
        if slice_has_celltype[sid]:
            panels.append(('slice', sid, 'celltype', f'Slice {sid} (original celltypes)'))
        elif has_knn_celltypes:
            panels.append(('slice', sid, 'celltype_knn', f'Slice {sid} (KNN celltypes)'))

    if include_ref:
        panels.append(('reference', -1, 'celltype', 'Reference (celltypes)'))

    if len(panels) > 0:
        n_panels = len(panels)
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

        for ax, (source, sid, ct_col, title) in zip(axes, panels):
            if source == 'reference':
                mask = combined_hvg.obs['_source'] == 'reference'
            else:
                mask = (combined_hvg.obs['_slice_id'] == sid) & (combined_hvg.obs['_source'] == 'query')

            cell_labels = combined_hvg.obs.loc[mask, ct_col].values
            umap_coords = X_umap_after[mask]

            for ct in all_celltypes_list:
                ct_mask = cell_labels == ct
                if ct_mask.sum() > 0:
                    ax.scatter(
                        umap_coords[ct_mask, 0], umap_coords[ct_mask, 1],
                        c=[ct_color_map[ct]], s=2, alpha=0.5, label=ct,
                    )
            ax.set_title(title)
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            ax.legend(markerscale=5, fontsize=7, framealpha=0.9, loc='best')

        plt.suptitle('Post-Harmony UMAP by Cell Type', fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print("  No cell type labels available for visualization.")
else:
    print("  No cell type labels available. Skipping cell type UMAP.")
```

---

## Parameter Guide

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `slice_ids` | `[0, 1]` | any 2+ | Which slices to integrate (user must specify) |
| `reference_path` | `None` | path | Optional reference scRNA-seq h5ad with `celltype` column |
| `use_knn_transfer` | `False` | bool | Transfer cell types from reference to unlabeled slices via KNN |
| `n_top_genes` | `2000` | 500-5000 | Highly variable genes for integration |
| `n_pcs` | `30` | 10-50 | Principal components for Harmony |
| `max_iter_harmony` | `20` | 5-50 | Harmony iterations |
| `n_knn` | `15` | 5-50 | Neighbors for KNN label transfer |

## Notes

- **No new slice created**: Harmony embeddings are stored back into each original slice's adata. Access via `session.get_slice(sid).adata.obsm['X_pca_harmony']`.
- **Reference is temporary**: If a reference h5ad is provided, it is used during integration and KNN transfer but is NOT stored in any slice. Only the query slices receive results.
- **KNN label transfer**: Uses `sklearn.neighbors.KNeighborsClassifier` with distance-weighted voting in Harmony-corrected PCA space. Confidence scores are stored in `obs['celltype_knn_confidence']`.
- **Cell type preservation**: If a slice already has celltype annotations, they are kept. KNN results are stored separately as `celltype_knn` for comparison.
- **Gene intersection**: `join='inner'` keeps only genes shared across all datasets. If reference has very different gene panel, the intersection may be small.
- **Before/after comparison**: Both pre-integration and post-integration UMAP coordinates are stored for visual comparison of batch correction quality.
