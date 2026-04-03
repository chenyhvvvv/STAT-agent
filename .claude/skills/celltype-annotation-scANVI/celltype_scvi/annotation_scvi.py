"""
Cell type annotation using scANVI (scvi-tools).

This module provides functionality to annotate spatial transcriptomics data
using a reference scRNA-seq dataset with known cell type labels.

Method: scANVI (single-cell ANnotation using Variational Inference)
- Integrates reference and query data using variational autoencoders
- Transfers cell type labels via semi-supervised learning
- Handles batch effects between technologies (scRNA-seq vs spatial)

Based on:
- Lopez et al., Nat Methods 2018 (SCVI)
- Xu et al., Mol Syst Biol 2021 (scANVI)
"""

from typing import Optional, Union
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc


def annotate_celltype_scvi(
    adata_spatial: ad.AnnData,
    adata_reference: ad.AnnData,
    label_key: str = 'celltype',
    n_latent: int = 30,
    max_epochs_scvi: Optional[int] = None,
    max_epochs_scanvi: int = 20,
    n_samples_per_label: int = 100,
    use_hvg: bool = True,
    n_hvg: int = 1000,
    seed: int = 0
) -> pd.Series:
    """
    Annotate cell types in spatial data using scANVI transfer learning.

    Parameters
    ----------
    adata_spatial : ad.AnnData
        Spatial transcriptomics data to annotate.
        Must have gene expression in .X
    adata_reference : ad.AnnData
        Reference scRNA-seq data with known cell types.
        Must have label_key column in .obs
    label_key : str, default='celltype'
        Column name in reference.obs containing cell type labels
    n_latent : int, default=30
        Number of latent dimensions in SCVI model
    max_epochs_scvi : Optional[int], default=None
        Maximum training epochs for SCVI (None = auto)
    max_epochs_scanvi : int, default=20
        Maximum training epochs for scANVI
    n_samples_per_label : int, default=100
        Number of samples per label for scANVI training
    use_hvg : bool, default=True
        Whether to use highly variable genes for large datasets
    n_hvg : int, default=1000
        Number of HVGs to use (if use_hvg=True and sufficient genes)
    seed : int, default=0
        Random seed for reproducibility

    Returns
    -------
    pd.Series
        Predicted cell type labels for spatial data.
        Index matches adata_spatial.obs.index

    Raises
    ------
    ImportError
        If scvi-tools is not installed
    ValueError
        If label_key not found in reference data
        If too few common genes between datasets

    Examples
    --------
    >>> import scanpy as sc
    >>> st_data = sc.read_h5ad('spatial.h5ad')
    >>> ref_data = sc.read_h5ad('reference.h5ad')
    >>> celltype_labels = annotate_celltype_scvi(st_data, ref_data)
    >>> st_data.obs['celltype'] = celltype_labels
    """

    # Import scvi-tools
    try:
        import scvi
        print(f"Using scvi-tools version: {scvi.__version__}")
    except ImportError:
        raise ImportError(
            "scvi-tools is required for celltype annotation. "
            "Install with: pip install scvi-tools"
        )

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set random seed
    scvi.settings.seed = seed

    # Validate inputs
    if label_key not in adata_reference.obs.columns:
        raise ValueError(
            f"Label key '{label_key}' not found in reference data. "
            f"Available columns: {list(adata_reference.obs.columns)}"
        )

    # Copy data to avoid modifying originals
    st_adata = adata_spatial.copy()
    sc_adata = adata_reference.copy()

    print(f"\nInput data:")
    print(f"  Spatial: {st_adata.n_obs:,} cells × {st_adata.n_vars:,} genes")
    print(f"  Reference: {sc_adata.n_obs:,} cells × {sc_adata.n_vars:,} genes")

    # Make gene names unique
    st_adata.var_names_make_unique()
    sc_adata.var_names_make_unique()

    # Find common genes
    common_genes = st_adata.var_names.intersection(sc_adata.var_names)
    print(f"  Common genes: {len(common_genes):,}")

    if len(common_genes) < 100:
        raise ValueError(
            f"Too few common genes ({len(common_genes)}). "
            f"Datasets may be incompatible or use different gene ID formats."
        )

    # Subset to common genes
    st_adata = st_adata[:, common_genes].copy()
    sc_adata = sc_adata[:, common_genes].copy()

    # Gene selection strategy
    if use_hvg and len(common_genes) >= n_hvg:
        print(f"\nSelecting {n_hvg} highly variable genes from reference...")

        # Normalize and log-transform reference for HVG selection
        sc_temp = sc_adata.copy()
        sc.pp.normalize_total(sc_temp, target_sum=1e4)
        sc.pp.log1p(sc_temp)

        # Find HVGs in reference
        sc.pp.highly_variable_genes(sc_temp, n_top_genes=n_hvg, flavor='seurat_v3')
        hvg_genes = sc_temp.var_names[sc_temp.var['highly_variable']]

        # Subset both datasets to HVGs
        st_adata = st_adata[:, hvg_genes].copy()
        sc_adata = sc_adata[:, hvg_genes].copy()

        print(f"  Selected {len(hvg_genes):,} HVGs")
    else:
        print(f"\nUsing all {len(common_genes):,} common genes")

    # Filter cells (reference only - spatial should already be filtered)
    sc.pp.filter_cells(sc_adata, min_counts=1)
    print(f"\nAfter filtering:")
    print(f"  Spatial: {st_adata.n_obs:,} cells × {st_adata.n_vars:,} genes")
    print(f"  Reference: {sc_adata.n_obs:,} cells × {sc_adata.n_vars:,} genes")

    # Add technology labels
    st_adata.obs["tech"] = "st"
    sc_adata.obs["tech"] = "sc"

    # Concatenate datasets
    print("\nCombining datasets...")
    adata_combined = ad.concat([sc_adata, st_adata])
    print(f"  Combined: {adata_combined.n_obs:,} cells")

    # Store raw counts
    adata_combined.layers["counts"] = adata_combined.X.copy()

    # Normalize and log-transform
    print("Normalizing...")
    sc.pp.normalize_total(adata_combined, target_sum=1e4)
    sc.pp.log1p(adata_combined)
    adata_combined.raw = adata_combined  # Keep full dimension safe

    # Setup SCVI
    print("\nSetting up SCVI model...")
    scvi.model.SCVI.setup_anndata(adata_combined, batch_key="tech")

    # Train SCVI
    print("Training SCVI model...")
    scvi_model = scvi.model.SCVI(
        adata_combined,
        n_layers=2,
        n_latent=n_latent
    )
    scvi_model.train(max_epochs=max_epochs_scvi)

    # Setup scANVI labels
    print("\nSetting up scANVI labels...")
    SCANVI_CELLTYPE_KEY = "celltype_scanvi"
    adata_combined.obs[SCANVI_CELLTYPE_KEY] = "Unknown"

    # Transfer labels from reference
    # CRITICAL: Use .loc to assign sc_adata labels to reference cells only
    sc_mask = adata_combined.obs["tech"] == "sc"
    adata_combined.obs.loc[sc_mask, SCANVI_CELLTYPE_KEY] = sc_adata.obs[label_key].values

    # Count labeled cells per type
    labeled_counts = adata_combined.obs[SCANVI_CELLTYPE_KEY].value_counts()
    print(f"  Labeled cell types: {len(labeled_counts)-1}")  # -1 for Unknown
    print(f"  Unlabeled cells: {labeled_counts.get('Unknown', 0):,}")

    # Train scANVI
    print("\nTraining scANVI model...")
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        adata=adata_combined,
        unlabeled_category="Unknown",
        labels_key=SCANVI_CELLTYPE_KEY,
    )
    scanvi_model.train(
        max_epochs=max_epochs_scanvi,
        n_samples_per_label=n_samples_per_label
    )

    # Get predictions
    print("\nPredicting cell types for spatial data...")
    SCANVI_LATENT_KEY = "X_scANVI"
    SCANVI_PREDICTION_KEY = "C_scANVI"

    adata_combined.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata_combined)
    adata_combined.obs[SCANVI_PREDICTION_KEY] = scanvi_model.predict(adata_combined)

    # Extract spatial predictions
    # CRITICAL: Don't use .copy() - just subset directly to preserve index alignment
    annotated_adata_st = adata_combined[adata_combined.obs["tech"] == "st"]
    celltype_predictions = annotated_adata_st.obs[SCANVI_PREDICTION_KEY]

    print(f"\n✓ Annotation complete!")
    print(f"  Predicted {celltype_predictions.nunique()} cell types")
    print(f"  Predictions shape: {len(celltype_predictions)}")
    print(f"  Original spatial shape: {adata_spatial.n_obs}")

    # Verify shape matches
    if len(celltype_predictions) != adata_spatial.n_obs:
        raise ValueError(
            f"Shape mismatch: predictions has {len(celltype_predictions)} cells "
            f"but original spatial data has {adata_spatial.n_obs} cells. "
            f"This may indicate filtering changed cell counts."
        )

    return celltype_predictions


def validate_annotation_inputs(
    adata_spatial: ad.AnnData,
    adata_reference: ad.AnnData,
    label_key: str = 'celltype'
) -> dict:
    """
    Validate inputs for celltype annotation.

    Parameters
    ----------
    adata_spatial : ad.AnnData
        Spatial data
    adata_reference : ad.AnnData
        Reference data
    label_key : str
        Column name for cell type labels

    Returns
    -------
    dict
        Validation results with keys:
        - 'valid': bool
        - 'errors': list of error messages
        - 'warnings': list of warning messages
        - 'info': dict with dataset statistics
    """
    errors = []
    warnings_list = []
    info = {}

    # Check spatial data
    if adata_spatial.n_obs == 0:
        errors.append("Spatial data has 0 cells")
    if adata_spatial.n_vars == 0:
        errors.append("Spatial data has 0 genes")

    # Check reference data
    if adata_reference.n_obs == 0:
        errors.append("Reference data has 0 cells")
    if adata_reference.n_vars == 0:
        errors.append("Reference data has 0 genes")

    # Check label column
    if label_key not in adata_reference.obs.columns:
        errors.append(f"Label key '{label_key}' not in reference.obs")
    else:
        n_types = adata_reference.obs[label_key].nunique()
        info['n_celltypes'] = n_types
        if n_types < 2:
            warnings_list.append(f"Reference has only {n_types} cell type(s)")

        # Check cells per type
        counts = adata_reference.obs[label_key].value_counts()
        min_cells = counts.min()
        if min_cells < 10:
            warnings_list.append(
                f"Some cell types have <10 cells (min: {min_cells}). "
                f"May affect annotation quality."
            )

    # Check common genes
    common = adata_spatial.var_names.intersection(adata_reference.var_names)
    info['n_common_genes'] = len(common)
    if len(common) < 100:
        errors.append(f"Only {len(common)} common genes. Minimum 100 required.")
    elif len(common) < 500:
        warnings_list.append(f"Only {len(common)} common genes. >500 recommended.")

    # Info
    info['spatial_cells'] = adata_spatial.n_obs
    info['spatial_genes'] = adata_spatial.n_vars
    info['reference_cells'] = adata_reference.n_obs
    info['reference_genes'] = adata_reference.n_vars

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings_list,
        'info': info
    }
