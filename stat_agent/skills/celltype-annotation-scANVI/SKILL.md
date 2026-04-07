---
name: celltype-annotation-scANVI
title: Cell Type Annotation with scANVI
slug: celltype-annotation-scanvi
description: Annotate cell types in spatial transcriptomics data using scANVI transfer learning from a reference scRNA-seq dataset. (Recommended for cell data)
filter_requirements:
  num_slices: 1
  modalities: [gene]
  data_levels: [cell]

prerequisites:
  - Annotated reference scRNA-seq dataset path (.h5ad file)
  - Cell type column name in the reference dataset (default celltype)
default_skill: true
---

# Cell Type Annotation with scANVI

Annotate cell types in spatial transcriptomics data using **scANVI** (single-cell ANnotation using Variational Inference), a transfer learning method that integrates a reference scRNA-seq dataset with known cell type labels.

## Workflow

### Stage 1: Load and Validate

```python
import scanpy as sc
from pathlib import Path
from celltype_scvi.annotation_scvi import annotate_celltype_scvi

# IMPORTANT: Specify the target slice (e.g. slice 0)
slice_0 = session.get_slice(0)

print("="*60)
print("STAGE 1: Load and Validate")
print("="*60)

# Load reference data (change to the reference path)
reference_path = '/path/to/reference.h5ad'
if not Path(reference_path).exists():
    raise FileNotFoundError(f"Reference file not found: {reference_path}")

ref_adata = sc.read_h5ad(reference_path)
print(f"✓ Reference loaded: {ref_adata.n_obs:,} cells, {ref_adata.n_vars:,} genes")

# Validate reference has celltype column
label_key = 'celltype'  # Or user-specified column
if label_key not in ref_adata.obs.columns:
    raise ValueError(
        f"Reference does not have '{label_key}' column. "
        f"Available columns: {list(ref_adata.obs.columns)}"
    )

print(f"✓ Spatial data: {slice_0.adata.n_obs:,} cells, {slice_0.adata.n_vars:,} genes")
print(f"✓ Reference cell types: {ref_adata.obs[label_key].nunique()}")
```

### Stage 2: Run scANVI Annotation

```python
print("\n" + "="*60)
print("STAGE 2: scANVI Annotation")
print("="*60)

try:
    celltype_predictions = annotate_celltype_scvi(
        adata_spatial=slice_0.adata,
        adata_reference=ref_adata,
        label_key=label_key,
        n_latent=30,
        max_epochs_scvi=None,   # Auto
        max_epochs_scanvi=20
    )

    print(f"\n✓ Annotation completed!")
    print(f"  Predicted {celltype_predictions.nunique()} cell types")

    # Show distribution
    print(f"\n  Cell type distribution:")
    counts = celltype_predictions.value_counts()
    for celltype, count in counts.head(10).items():
        print(f"    {celltype}: {count:,} cells ({count/len(celltype_predictions):.1%})")
    if len(counts) > 10:
        print(f"    ... and {len(counts)-10} more cell types")

except Exception as e:
    print(f"\n❌ Annotation failed: {e}")
    raise
```

### Stage 3: Store Results

```python
print("\n" + "="*60)
print("STAGE 3: Storing Results")
print("="*60)

# Direct assignment - indices match
slice_0.adata.obs['celltype'] = celltype_predictions

print(f"✓ Added 'celltype' column to slice_0.adata.obs")
print(f"  Shape: {slice_0.adata.shape}")
print(f"  Celltype unique values: {slice_0.adata.obs['celltype'].nunique()}")

# Verify
assert 'celltype' in slice_0.adata.obs.columns, "Failed to add celltype column"
assert slice_0.adata.obs['celltype'].notna().all(), "Celltype contains NA values"

print("\n✓ Annotation complete! Celltype column added successfully.")
print("\nYou can now:")
print("  - Reload cell overlay in the web interface to see colored cells")
print("  - Analyze cell type distributions")
print("  - Compare cell types across ROIs")
```
