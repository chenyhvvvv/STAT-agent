---
name: celltype-annotation-fast
title: Fast Cell Type Annotation (Clustering + LLM)
slug: celltype-annotation-fast
description: Annotate cell types using unsupervised clustering, marker genes, and LLM-based annotation. Fast alternative to reference-based methods.

filter_requirements:
  modalities: [gene]
  data_levels: [cell]

prerequisites:
  - "Tissue type information (e.g., breast cancer, brain cortex, liver)"
default_skill: true
---

# Fast Cell Type Annotation with Clustering + LLM

Annotate cell types in spatial transcriptomics data using an unsupervised approach:
1. **Leiden clustering** to identify cell populations
2. **Marker gene identification** via differential expression
3. **LLM-based annotation** using tissue context

---

## Workflow

### Stage 1: Validation

```python
# IMPORTANT: First specific the slice needed to be annotate (Here, use slice 0 as example)
slice_0 = session.get_slice(0)

print("="*60)
print("STAGE 1: Validation")
print("="*60)

# Import validation function
from celltype_fast.annotation_clustering import validate_annotation_inputs

# Validate inputs
validation = validate_annotation_inputs(
    adata_spatial=slice_0.adata, # Input: slice adata
    tissue_type='breast cancer'
)

if not validation['valid']:
    print("❌ Validation failed:")
    for error in validation['errors']:
        print(f"  - {error}")
    raise ValueError("Input validation failed")

if validation['warnings']:
    print("⚠️  Warnings:")
    for warning in validation['warnings']:
        print(f"  - {warning}")

print(f"\n✓ Validation passed")
print(f"  Cells: {validation['info']['spatial_cells']:,}")
print(f"  Genes: {validation['info']['spatial_genes']:,}")
print(f"  Tissue: {validation['info']['tissue_type']}")
```

### Stage 2: Define LLM Annotation Function

```python
print("\n" + "="*60)
print("STAGE 2: Setup LLM Annotation")
print("="*60)
# No additional configuration needed
# Import session LLM annotator
from celltype_fast.annotation_clustering import annotate_with_session_llm

# Define annotation function (uses session's configured LLM)
def llm_annotator(markers_dict, tissue_type):
    """Use the LLM you configured during session initialization."""
    return annotate_with_session_llm(
        markers_dict=markers_dict,
        tissue_type=tissue_type,
        session=session  # Uses session's API key and model
    )

print("✓ LLM annotator configured (using session LLM)")
print(f"  Model: {session.llm_config.get('model', 'unknown')}")
```

---

### Stage 3: Run Clustering and Annotation

```python
print("\n" + "="*60)
print("STAGE 3: Clustering and Annotation")
print("="*60)

# Import main function
from celltype_fast.annotation_clustering import annotate_celltype_clustering

try:
    # Run annotation
    celltype_predictions = annotate_celltype_clustering(
        adata_spatial=slice_0.adata,
        tissue_type='breast cancer',  # CRITICAL: Specify tissue type!
        llm_function=llm_annotator,
        resolution=0.5,         # Clustering resolution (0.3-1.0)
        n_top_genes=10,         # Top marker genes per cluster
        min_cluster_size=10,    # Min cells per cluster
        preprocess=True         # Run preprocessing (normalize, HVG, PCA, neighbors)
    )

    print(f"\n✓ Annotation completed successfully!")
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

### Stage 4: Store Results

```python
# CRITICAL: Store celltype in session.adata.obs
# The returned predictions have the correct index alignment
print("\n" + "="*60)
print("STAGE 4: Storing Results")
print("="*60)

# Direct assignment - the indices match
slice_0.adata.obs['celltype'] = celltype_predictions

print(f"✓ Added 'celltype' column to session.adata.obs")
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
---

## Tissue Type Context

The `tissue_type` parameter is **critical** for good annotation. It provides biological context to the LLM. Please specific it based on the data information.
